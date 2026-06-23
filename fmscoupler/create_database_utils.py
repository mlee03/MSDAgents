"""
create_database_utils.py

Shared configuration, embedding model, splitters, and helper functions
used by create_readme_collection.py, create_code_module_collection.py,
and create_unified_database.py.
"""

import logging

from pydantic import BaseModel
from pymilvus import connections, utility

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

MILVUS_HOST      = "localhost"
MILVUS_PORT      = 19530
HUGGINGFACE_MODEL = "sentence-transformers/all-mpnet-base-v2"
DENSE_DIM        = 768   # output dimension of all-mpnet-base-v2
CHUNK_OVERLAP    = 100   # character overlap between sub-chunks
MAX_TEXT_LEN     = 65535
MAX_ID_LEN       = 512

# ---------------------------------------------------------------------------
# Embedding model 
# ---------------------------------------------------------------------------

dense_ef = HuggingFaceEmbeddings(
    model_name=HUGGINGFACE_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
tokenizer = dense_ef._client.tokenizer
MAX_TOKEN_LENGTH = tokenizer.model_max_length

# ---------------------------------------------------------------------------
# Markdown splitters
# ---------------------------------------------------------------------------

# h1/h2/h3 — used by the narrative docs parser
h123_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=True,
    return_each_line=False,
)

# h1/h2 and h3 — used by the code-module parser (two-pass)
h12_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2")],
    strip_headers=True,
    return_each_line=False,
)
h3_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("###", "h3")],
    strip_headers=True,
    return_each_line=False,
)

# ---------------------------------------------------------------------------
# ID 
# ---------------------------------------------------------------------------

def make_id(meta: dict) -> str:
    """Build a hierarchical chunk ID: h1/h2/h3 (omits missing levels)."""
    return "/".join(meta[h] for h in ("h1", "h2", "h3") if meta.get(h))

# ---------------------------------------------------------------------------
# Metadata model
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    source: str
    name: str
    parent: str
    ichunk: int = 1
    datatype: str 

# ---------------------------------------------------------------------------
# Chunk helper
# ---------------------------------------------------------------------------

def check_chunk_length(chunk, end_in_error=False):
    """Check the token length of a section and print a warning if it exceeds MAX_TOKEN_LENGTH."""
    ntokens = len(tokenizer.tokenize(chunk))
    if ntokens > MAX_TOKEN_LENGTH:
        message = f"Section token length of {ntokens} exceeds the maximum of {MAX_TOKEN_LENGTH}."
        if end_in_error:
            raise RuntimeError(message)
        else:
            print(message)
    return ntokens

# ---------------------------------------------------------------------------
# Milvus helpers
# ---------------------------------------------------------------------------

def connect_vectorstore(collection_name: str) -> Milvus:
    """Return a Milvus handle attached to an existing collection."""
    return Milvus(
        embedding_function=dense_ef,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        collection_name=collection_name,
    )


def remove_collection_if_exists(collection_name: str) -> None:
    """Drop a Milvus collection if it already exists."""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"[milvus] Dropped existing collection: {collection_name}")


def create_milvus_database(documents: list, ids: list, collection_name: str) -> None:
    """Embed documents and (re)build a Milvus hybrid collection."""
    print(f"\n[milvus] Connecting to {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"[milvus] Collection  : {collection_name}")
    print(f"[milvus] Embedding   : {HUGGINGFACE_MODEL}  ({DENSE_DIM}-dim dense + BM25 sparse)")
    remove_collection_if_exists(collection_name)
    Milvus.from_documents(
        documents=documents,
        ids=ids,
        embedding=dense_ef,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        collection_name=collection_name,
        drop_old=False,
    )
    print(f"\n[milvus] Ingestion complete -- {len(documents)} documents stored.")



# ---------------------------------------------------------------------------
# Test retrieval
# ---------------------------------------------------------------------------

def test_collection(
    collection_name: str,
    tests: list[tuple[str, str | None]],
    log_file: str,
) -> None:
    """
    Log every stored chunk with token counts, then run test queries.
    Results are written to log_file and summarised on stdout.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        filename=log_file, filemode="w", force=True,
    )
    log = logging.getLogger(__name__)

    vs    = connect_vectorstore(collection_name)
    vs.col.load()
    total = vs.col.query(expr="", output_fields=["count(*)"])[0]["count(*)"]
    log.info(f"Collection : {collection_name}")
    log.info(f"Schema     : {vs.col.schema}\n***")
    log.info(f"Total entities: {total}\n***")

    for result in vs.col.query(expr="", output_fields=["*"], limit=total):
        token_count = len(tokenizer.encode(result["text"]))
        if token_count > MAX_TOKEN_LENGTH:
            print(f"WARNING: '{result.get('name','?')}' exceeds max token length "
                  f"({token_count} > {MAX_TOKEN_LENGTH}).")
        log.info(f"name   : {result.get('name') or result.get('source', 'unknown')}")
        log.info(f"tokens : {token_count}")
        log.info(f"source : {result.get('source', '')}")
        log.info(f"parent : {result.get('parent', '')}")
        log.info(f"ichunk : {result.get('ichunk', '')}")
        log.info(result["text"])
        log.info("***\n")

    print("\n" + "=" * 72)
    print(f"\n[done] Full log written to: {log_file}")