from abc import ABC, abstractmethod
import re
from pprint import pformat

from pymilvus import DataType, Function, FunctionType, MilvusClient
from pymilvus import AnnSearchRequest, WeightedRanker
from pymilvus import LexicalHighlighter

from shared.embeddings import (
    EmbeddingClass, 
    TokenizerClass,
    SentenceTransformerEmbedding,
    SentenceTransformerTokenizer
)
from shared.collection_data import (
    schema_metadata_fields, 
    schema_vector_fields,
    indexes,
    sparse_vector_search_function,
    CollectionData
)

class ClientClass(ABC):

    @abstractmethod
    def connect(self):
        pass

class RetrieverClass(ABC):

    @abstractmethod
    def retrieve(self, query: str):
        pass


class Client(ClientClass):

    def __init__(self, 
                 collection_name: str, 
                 milvus_host: str="localhost", 
                 milvus_port: int = 19530,
                 connect: bool = True,
                 schema = None,
                 index_params = None,
                 embedding: EmbeddingClass = SentenceTransformerEmbedding(),
                 tokenizer: TokenizerClass = SentenceTransformerTokenizer()
                 
    ):

        self.collection_name = collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.uri = f"http://{self.milvus_host}:{self.milvus_port}"

        self.schema = schema
        self.index_params = index_params
        self.collection = None

        self.embedding = embedding
        self.tokenizer = tokenizer

        self.client = None

        if connect:
            self.connect()
        

    def connect(self):

        """
        Connect to Milvus
        """

        self.client = MilvusClient(uri=self.uri)  
        print(f"Connected to Milvus at {self.uri}")
        print(f"Collections found: {self.client.list_collections()}")


class newCollection(Client):

    def __init__(self, 
                 collection_name: str, 
                 milvus_host: str="localhost", 
                 milvus_port: int = 19530,
                 connect: bool = True,
                 schema = None,
                 index_params = None,
                 embedding: EmbeddingClass = SentenceTransformerEmbedding(),
                 tokenizer: TokenizerClass = SentenceTransformerTokenizer()

    ):
        super().__init__(collection_name=collection_name, milvus_host=milvus_host, milvus_port=milvus_port,
                         connect=connect, schema=schema, index_params=index_params, embedding=embedding, tokenizer=tokenizer)
        
    def create_schema(self):
        
        """
        Create default schema in accordance to shared/metadata
        """

        self.schema = self.client.create_schema(enable_dynamic_field=True)

        for field_name, field_info in schema_metadata_fields.items():
            self.schema.add_field(**field_info)

        for field_name, field_info in schema_vector_fields.items():
            self.schema.add_field(**field_info)        
        
        self.schema.add_function(sparse_vector_search_function)


    def add_index(self):

        """
        Add index to the collection
        """
        
        self.index_params = self.client.prepare_index_params()
        for index, index_info in indexes.items():
            self.index_params.add_index(**index_info)


    def create_collection(self):
        
        """
        Create collection
        If collection exists, overwrite
        """
        
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        if self.schema is None: self.create_schema()
        if self.index_params is None: self.add_index()
        
        self.client.create_collection(
            collection_name=self.collection_name, 
            schema=self.schema,
            index_params=self.index_params
        )


    def add_data(self, data: list = None):

        """
        Add data to collection
        """

        data_list = []
        for i, datum in enumerate(data, start=1):
            data_dict = datum.dict()
            print(i, data_dict["name"], data_dict["sourcefile"], data_dict["ichunk"], data_dict["chunks"])
            data_dict["dense_vector"] = self.embedding.encode(data_dict["text"])            
            data_list.append(data_dict)
        
        self.client.insert(self.collection_name, data=data_list)
        
    
    def test_collection(self, logfile: str = None):

        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
        else:
            raise RuntimeError(f"Collection '{self.collection_name}' does not exist.")

        if logfile is None:
            logfile = f"{self.collection_name}.log"

        data = []
        batch_size = 1000

        # Use query_iterator to stream all rows without manual offset paging.
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            batch_size=batch_size,
            limit=-1,
            filter="id >= 0",
            output_fields=["*"],
        )

        while True:
            rows = iterator.next()
            if not rows:
                iterator.close()
                break
            data.extend(rows)

        with open(logfile, "w", encoding="utf-8") as f:
            f.write(f"Collection: {self.collection_name}\n")
            f.write(pformat(self.client.describe_collection(self.collection_name)))
            f.write("\n *** \n\n")
            for row in data:
                f.write(f"ntokens:    {len(self.tokenizer.tokenize(row['text']))}\n")
                for schema_key in schema_metadata_fields:
                    f.write(f"{schema_key}: {row[schema_key]}\n")

        return data

    
class MilvusRetriever():

    def __init__(self, client: Client):
        self.client = client
        self.retrieve_limit = 10
        self.retrieve = self.hybrid_search


    def simple_retrieve(self, query: str, filter_file: bool = True, filter_text: bool = True):
        
        search_options = {
            "anns_field": "dense_vector",
            "limit": self.retrieve_limit,
            "output_fields": ["*"],
        }

        filters = self._get_filters(query, filter_file=filter_file, filter_text=filter_text)
        if filters: search_options["filter"] = filters

        returned_fields = self.client.client.search(
            self.client.collection_name,
            data = [self.client.embedding.encode(query)],
            **search_options
        )

        return self._format_output(returned_fields)


    def hybrid_search(self, query: str):
        dense_request = AnnSearchRequest(
            data=[self.client.embedding.encode(query)], 
            anns_field="dense_vector",
            limit=self.retrieve_limit,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            expr=self._get_filters(query)
        )
        sparse_request = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            limit=self.retrieve_limit,
            param={"metric_type": "BM25"},
            expr=self._get_filters(query)
        )
        ranker = Function(
            name="rrf",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={"reranker": "rrf", "k": 100}
        )
        returned = self.client.client.hybrid_search(
            collection_name=self.client.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=self.retrieve_limit,
            output_fields=["*"],
        )
        return(self._format_output(returned))


    def _get_filters(self, query: str, filter_file: bool = True, filter_text: bool = True):

        filters = ""
        if filter_file:
            filters = self._filter_file(query)
                    
        if filter_text:
            filtered_text = self._filter_text(query)
            if filtered_text:
                if filters:
                    filters += " and " + filtered_text
                else:
                    filters = filtered_text

        return filters


    def _filter_file(self, query: str):

        keyfiles = {
            "atm_land_ice_flux_exchange_mod": "atm_land_ice_flux_exchange_mod.md",
            "atmos_ocean_dep_fluxes_calc_mod": "atmos_ocean_dep_fluxes_calc_mod.md",
            "atmos_ocean_fluxes_calc_mod": "atmos_ocean_fluxes_calc_mod.md",
            "flux_exchange_mod": "flux_exchange_mod.md",
            "full_coupler_mod": "full_coupler_mod.md",
            "ice_ocean_flux_exchange_mod": "ice_ocean_flux_exchange_mod.md",
            "land_ice_flux_exchange_mod": "land_ice_flux_exchange_mod.md"
        }

        # add method to correct typo

        filters = re.findall(r'@(\w+)', query)
        if filters:
            filter_strings = [
                f"sourcefile like '{keyfiles[word]}'" for word in filters if word in keyfiles
            ]
            return " and ".join(filter_strings) if filter_strings else ""
        return ""


    def _filter_text(self, query: str):

        filters = re.findall(r'/(\w+)', query)
        if filters:
            filter_strings = [f"name like '%{filter_word}%'" for filter_word in filters]
            #filter_strings = [f"TEXT_MATCH(text, '{filter_word}')" for filter_word in filters]
            return " and ".join(filter_strings) if filter_strings else ""
        return ""

    def _format_output(self, retrieved: list):
        data = []
        for returned_field in retrieved[0]:
            datum = {"score": returned_field["distance"]}            
            for key in returned_field["entity"]:
                datum[key] = returned_field["entity"][key]
            data.append(datum)
        return data


if __name__ == "__main__":
    retriever = MilvusRetriever(client=None)  # Replace `None` with an actual `Client` instance if available
    retriever._filter_file("This is a test @land_ice_flux_exchange_mod")
