"""
Ingest FMSCoupler narrative documentation into Milvus.

Source : full/docs/*.md
Collection : FMSCouplerDocs

Requires Milvus standalone on localhost:19530.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from create_database_utils import (
    dense_ef, tokenizer, MAX_TOKEN_LENGTH, CHUNK_OVERLAP,
    h123_splitter,
    make_id, ChunkMetadata,
    create_milvus_database, test_collection, check_chunk_length,
    HUGGINGFACE_MODEL, MILVUS_HOST, MILVUS_PORT,
)

# ---------------------------------------------------------------------------
# Collection 
# ---------------------------------------------------------------------------

COLLECTION_NAME = "FMSCouplerDocs"
LOG_FILE        = "ingest_docs.log"

# ---------------------------------------------------------------------------
# Documentation files
# ---------------------------------------------------------------------------

DOC_FILES = [
    "README.md",
    "FLUX.md",
    "AtmosDataType.md",
    "IceDataType.md",
    "LandDataType.md",
    "OceanPublicType.md",
    "OceanStateType.md",
    "AtmosIceBoundaryType.md",
    "AtmosLandBoundaryType.md",
    "IceOceanBoundaryType.md",
    "LandIceAtmosBoundaryType.md",
    "OceanIceBoundaryType.md",
    "IceOceanDriverType.md",
]

# ---------------------------------------------------------------------------
# Chunk splitter
# ---------------------------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=MAX_TOKEN_LENGTH * 3,
    chunk_overlap=CHUNK_OVERLAP,
)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_doc(filepath: Path) -> tuple[list[Document], list[str]]:
    """Parse one *.md narrative doc into Documents with IDs."""

    documents: list[Document] = []
    ids: list[str] = []

    for section in h123_splitter.split_text(filepath.read_text(encoding="utf-8")):
        content = section.page_content.strip()        
        section_id = make_id(section.metadata)
        splitted_content = splitter.split_text(content)
        add_chunk = False if len(splitted_content) == 1 else True
        for ichunk, chunk_text in enumerate(splitted_content, start=1):
            chunk_text = chunk_text.strip()
            name = f"{section_id}/chunk{ichunk}" if add_chunk else f"{section_id}"
            metadata = ChunkMetadata(
                source=filepath.name,
                name=name,
                parent=section_id,
                datatype="readme",
                ichunk=ichunk,
            )
            documents.append(Document(page_content=chunk_text, metadata=metadata.model_dump()))
            ids.append(metadata.name)

    return documents, ids

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(docs_dir: Path|str, create_database: bool = False) -> tuple[list[Document], list[str]] | None:
    """Parse all documentation files.
    
    Args:
        docs_dir: Directory containing the narrative documentation markdown files.
        create_database: If True, ingest into FMSCouplerDocs Milvus collection.
                        If False, return the documents and ids instead.
    
    Returns:
        If create_database is False: (all_documents, all_ids)
        If create_database is True: None
    """
    all_documents: list[Document] = []
    all_ids: list[str] = []

    filepaths = [Path(docs_dir) / filename for filename in DOC_FILES]
    for filepath in filepaths:
        if not filepath.exists():
            raise FileNotFoundError(f"  {filepath.name}  MISSING")

    print(f"parse {len(DOC_FILES)}\n")
    for filepath in filepaths:
        docs, ids = parse_doc(filepath)
        all_documents.extend(docs)
        all_ids.extend(ids)
        print(f"  {filepath.name}  {len(docs)} documents")

    print(f"\n[chunk] Total: {len(all_documents)} documents")
    
    if create_database:
        create_milvus_database(all_documents, all_ids, COLLECTION_NAME)
    else:
        return all_documents, all_ids

# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------

TESTS = [
    ("What are the model component state types in the full coupler?",      None),
    ("How does the fast loop implicit tridiagonal diffusion scheme work?", None),
    ("What are the fields in atmos_data_type for radiative fluxes?",      "datatype"),
    ("What fields does ice_ocean_boundary_type pass from ice to ocean?",  "boundary_type"),
    ("How is MPI PE layout configured for atmosphere and ocean?",          "overview"),
    ("What controls concurrent radiation in OpenMP threading?",            "overview"),
    ("How is the model start time determined from coupler.res?",           None),
    ("What is the difference between fast ice and slow ice physics?",      None),
    ("What are the REGRID, REDIST, and DIRECT flux transfer modes?",      "flux_exchange"),
    ("What does the ice_ocean_driver_type control structure do?",         "driver_type"),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build(docs_dir=Path("full/docs"), create_database=True)
    test_collection(COLLECTION_NAME, TESTS, LOG_FILE)
    print(f"""
To use this collection in a chatbot:

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import BM25BuiltInFunction, Milvus

    dense_ef = HuggingFaceEmbeddings(model_name="{HUGGINGFACE_MODEL}",
                                     encode_kwargs={{"normalize_embeddings": True}})
    db = Milvus(embedding_function=dense_ef, builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={{"host": "{MILVUS_HOST}", "port": {MILVUS_PORT}}},
                collection_name="{COLLECTION_NAME}")
    retriever = db.as_retriever(search_kwargs={{"k": 4}})
""")
