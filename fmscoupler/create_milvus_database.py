"""
Create the coupler database with Milvus - hybrid (dense + sparse BM25) collection.
Uses langchain_milvus for collection creation, langchain_huggingface for dense embeddings,
and pymilvus BM25EmbeddingFunction for sparse vectors.
The resulting collection can be searched with both semantic and keyword queries
combined via RRFRanker or WeightedRanker.

Requires: pip install langchain-milvus langchain-huggingface pymilvus[model]
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus

import doxygen_xml_parser_chunking

MODULEFILES = {
    "atm_land_ice_flux_exchange_mod": "namespaceatm__land__ice__flux__exchange__mod.xml",
    "atmos_ocean_dep_fluxes_calc_mod": "namespaceatmos__ocean__dep__fluxes__calc__mod.xml",
    "atmos_ocean_fluxes_calc_mod": "namespaceatmos__ocean__fluxes__calc__mod.xml",
    "flux_exchange_mod": "namespaceflux__exchange__mod.xml",
    "full_coupler_mod": "namespacefull__coupler__mod.xml",
    "ice_ocean_flux_exchange_mod": "namespaceice__ocean__flux__exchange__mod.xml",
    "land_ice_flux_exchange_mod": "namespaceland__ice__flux__exchange__mod.xml",
}

READMES = [
    "full/docs/README.md",
    "full/docs/FLUX.md",
    "full/docs/AtmosDataType.md",
    "full/docs/AtmosIceBoundaryType.md",
    "full/docs/AtmosLandBoundaryType.md",
    "full/docs/IceDataType.md",
    "full/docs/IceOceanBoundaryType.md",
    "full/docs/IceOceanDriverType.md",
    "full/docs/LandDataType.md",
    "full/docs/LandIceAtmosBoundaryType.md",
    "full/docs/OceanIceBoundaryType.md",
    "full/docs/OceanPublicType.md",
    "full/docs/OceanStateType.md",
]

COLLECTION_NAME = "FMSCouplerHybridLangchain"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
HUGGINGFACE_MODEL = "sentence-transformers/all-mpnet-base-v2"  
DENSE_DIM = 768 # output dimension of all-mpnet-base-v2
MAX_TEXT_LEN = 65535 # Milvus VARCHAR max
MAX_ID_LEN = 512

dense_ef = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL, model_kwargs={"device": "cpu"})
tokenizer = dense_ef._client.tokenizer
MAX_TOKEN_LENGTH = tokenizer.model_max_length


def _setup_logging(log_file: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s", filename=log_file, filemode="w", force=True)
    return logging.getLogger(__name__)


def _connect_vectorstore() -> Milvus:
    return Milvus(
        embedding_function=dense_ef,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        collection_name=COLLECTION_NAME,
    )


def xml_to_markdown():

    readmes = []

    #hack
    if Path("./docs/xml/full_2flux__exchange_8_f90.xml").exists():
        print("Renaming flux__exchange_8_f90.xml to full_2flux__exchange_8_f90.xml...")
        Path("./docs/xml/full_2flux__exchange_8_f90.xml").rename("./docs/xml/flux__exchange_8_f90.xml")

    for module, xmlfile in MODULEFILES.items():
        print(f"Processing {module}...")
        modxml = doxygen_xml_parser_chunking.ModuleBodyDocument(xmldir="./docs/xml", xmlfile=xmlfile)
        modxml.document_module_variables()
        modxml.document_procedures()
        readmes.append(modxml.write_markdown())
    return readmes


def document_variable(section, this_id: str, markdownfile: str) -> Document:

    ntokens = len(tokenizer.tokenize(section.page_content))
    if ntokens > MAX_TOKEN_LENGTH:
        raise RuntimeError(f"tokens for {this_id}: {ntokens} > max {MAX_TOKEN_LENGTH}.")
    section.metadata = {"source": markdownfile, "name": this_id, "parent": this_id, "ichunk": 1}
    return section


def document_procedure(section, this_id: str, markdownfile: str) -> tuple[list[Document], list[str]]:

    documents: list[Document] = []
    ids: list[str] = []
    ichunk = 1

    subroutine_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("###", "h3")], strip_headers=True
    )

    for subsection in subroutine_splitter.split_text(section.page_content):

        parent = this_id
        this_subsection = subsection.metadata.get("h3", "")
        subsection_id = f"{this_id}::{this_subsection}"

        ntokens = len(tokenizer.tokenize(subsection.page_content))
        if ntokens > MAX_TOKEN_LENGTH:
            print(f"{subsection_id} has {ntokens} tokens, chunking.")

        if this_subsection == "flowchart":
            separators = ["(?=Step \\d+:)"]
        elif this_subsection == "arguments":
            separators = ["argument"]
        elif this_subsection == "description":
            separators = ["\n"]
        elif this_subsection == "intro":
            separators = ["\n"]
        else:
            raise RuntimeError(f"Unexpected subsection '{this_subsection}' in {this_id}.")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators, chunk_size=1000, chunk_overlap=0, is_separator_regex=True
        )
        chunked = text_splitter.split_documents([subsection])
        for chunk in chunked:
            chunk_id = f"{subsection_id}::chunk{ichunk}"
            chunk.metadata = {"source": markdownfile, "name": chunk_id, "parent": parent, "ichunk": ichunk}
            documents.append(chunk)
            ids.append(chunk_id)
            ichunk += 1

    return documents, ids


def parse_code_mds() -> tuple[list[Document], list[str]]:
    """Generate list of Documents and ids from the markdown files."""

    documents: list[Document] = []
    ids: list[str] = []

    top_markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")], strip_headers=True
    )

    markdownfiles = xml_to_markdown()
    for markdownfile in markdownfiles:
        print(f"Processing {markdownfile}...")

        readmefile = Path(markdownfile)
        if not readmefile.exists():
            raise FileNotFoundError(f"README file not found: {markdownfile}")

        sections = top_markdown_splitter.split_text(readmefile.read_text(encoding="utf-8"))        
        
        for section in sections[1:]:
            
            h1 = section.metadata.get("h1", "")
            h2 = section.metadata.get("h2", "")
            this_id = f"{h1}::{h2}"                
            
            if "variable" in h2:
                documents.append(document_variable(section, this_id, markdownfile))
                ids.append(this_id)
            
            if "subroutine" in h2 or "function" in h2:
                sub_docs, sub_ids = document_procedure(section, this_id, markdownfile)
                documents.extend(sub_docs)
                ids.extend(sub_ids)

    return documents, ids


def parse_readme_mds():
    """Generate list of Documents and ids from the README files."""

    documents: list[Document] = []
    ids: list[str] = []

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=True,
    )

    for readme in READMES:
        print(f"Processing {readme}...")

        readmefile = Path(readme)
        if not readmefile.exists():
            raise FileNotFoundError(f"README file not found: {readme}")

        sections = markdown_splitter.split_text(readmefile.read_text())

        for section in sections:
            this_id = "::".join([section.metadata[h] for h in ["h1", "h2", "h3"] if h in section.metadata])

            ntokens = len(tokenizer.tokenize(section.page_content))
            if ntokens > MAX_TOKEN_LENGTH:
                print(f"{this_id} has {ntokens} tokens, chunking.")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n"], chunk_size=1000, chunk_overlap=100, is_separator_regex=False
                )
                chunked = text_splitter.split_documents([section])
                for ichunk, chunk in enumerate(chunked, start=1):
                    chunk_id = f"{this_id}::chunk{ichunk}"
                    chunk.metadata = {"source": readme, "name": chunk_id, "parent": this_id, "ichunk": ichunk}
                    documents.append(chunk)
                    ids.append(chunk_id)

    return documents, ids


def build_collection() -> None:
    """
    Build a hybrid Milvus collection on standalone Milvus server.
    The collection stores both dense (semantic) and sparse (BM25) vectors,
    enabling hybrid search via RRFRanker or WeightedRanker.
    """

    code_documents, code_ids = parse_code_mds()
    readme_docs, readme_ids = parse_readme_mds()

    all_documents = code_documents + readme_docs
    all_ids = code_ids + readme_ids

    vectorstore = Milvus.from_documents(
        documents=all_documents,
        ids=all_ids,
        embedding=dense_ef,                      
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        collection_name=COLLECTION_NAME,
        drop_old=True,
        #enable_dynamic_field=True,
    )

    print(f"Created hybrid collection '{COLLECTION_NAME}' with {len(code_documents) + len(readme_docs)} documents.")


def test_collection(query: str = "flux exchange") -> None:
    """Test the hybrid collection: log collection info, token counts, and run searches."""

    log = _setup_logging("test_collection_hybrid.log")

    # Connect to the existing collection without rebuilding it
    vectorstore = _connect_vectorstore()
    
    vectorstore.col.load()
    total = vectorstore.col.query(expr="", output_fields=["count(*)"])[0]["count(*)"]

    log.info(f"Collection schema: {vectorstore.col.schema}\n***")
    log.info(f"Collection '{COLLECTION_NAME}' exists with {total} entities.\n***")

    # Log token counts for every stored document
    results = vectorstore.col.query(expr= "", output_fields=["*"], limit=total)

    tokenizer = dense_ef._client.tokenizer
    for result in results:
        token_count = len(tokenizer.encode(result["text"]))
        if token_count > tokenizer.model_max_length:
            print(f"Document '{result['name']}' exceeds max token length with {token_count} tokens.")
        name = result.get("name") or result.get("source", "unknown")
        log.info(f"{name}")
        log.info(f"tokens: {token_count}")
        log.info(f"source: {result['source']}")
        log.info(f"parent: {result['parent']}")
        log.info(f"ichunk: {result['ichunk']}")
        log.info(result["text"])
        log.info("***\n")

if __name__ == "__main__":
    build_collection()
    test_collection()

