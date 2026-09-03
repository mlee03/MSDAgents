from pydantic import BaseModel, ConfigDict
from numpy.typing import NDArray

from pymilvus import DataType, Function, FunctionType

schema_metadata_fields = {
    "id": {"field_name": "id", "datatype": DataType.INT64, "is_primary": True, "auto_id": True},
    "name": {"field_name": "name", "datatype": DataType.VARCHAR, "max_length": 65535},    
    "sourcefile": {"field_name": "sourcefile", "datatype": DataType.VARCHAR, "max_length": 65535},
    "is_chunked": {"field_name": "is_chunked", "datatype": DataType.BOOL},
    "ichunk": {"field_name": "ichunk", "datatype": DataType.INT32},
    "chunks": {"field_name": "chunks", "datatype": DataType.ARRAY, "element_type": DataType.INT32, "max_capacity": 25},
    "text": {"field_name": "text", "datatype": DataType.VARCHAR, "enable_analyzer": True, "enable_match": True, "max_length": 65535},
}

schema_vector_fields = {
    "dense_vector": {"field_name": "dense_vector", "datatype": DataType.FLOAT_VECTOR, "dim": 384},
    "sparse_vector": {"field_name": "sparse_vector", "datatype": DataType.SPARSE_FLOAT_VECTOR}
}

sparse_vector_search_function = Function(
    name="text_bm25_emb", 
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse_vector"]
)

indexes = {
    "dense_vector": {"field_name": "dense_vector", "index_type": "AUTOINDEX", "metric_type": "COSINE"},
    "sparse_vector": {"field_name": "sparse_vector", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
}

class CollectionData(BaseModel):

    name: str
    sourcefile: str
    is_chunked: bool
    ichunk: int = 1
    chunks: list = [1]
    text: str

