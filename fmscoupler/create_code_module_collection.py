"""
ingest_code.py

Ingest FMSCoupler Doxygen-generated module documentation into Milvus.

Source : coupler/*_mod.md  (7 files)
Collection : FMSCouplerCode

Usage
-----
    python ingest_code.py

Requires Milvus standalone on localhost:19530.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from create_database_utils import (
    dense_ef, tokenizer, MAX_TOKEN_LENGTH, CHUNK_OVERLAP,
    h12_splitter, h3_splitter,
    ChunkMetadata, create_milvus_database, test_collection,
    check_chunk_length,
    HUGGINGFACE_MODEL, MILVUS_HOST, MILVUS_PORT,
)

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

COLLECTION_NAME = "FMSCouplerCode"
LOG_FILE        = "ingest_code.log"

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

CODE_MODULE_FILES = [
    "full_coupler_mod.md",
    "flux_exchange_mod.md",
    "atm_land_ice_flux_exchange_mod.md",
    "atmos_ocean_fluxes_calc_mod.md",
    "atmos_ocean_dep_fluxes_calc_mod.md",
    "ice_ocean_flux_exchange_mod.md",
    "land_ice_flux_exchange_mod.md",
]

# Text splitters keyed by h3 subsection type (used in add_chunk)
SUBSECTION_SPLITTERS = {
    "flowchart": RecursiveCharacterTextSplitter(
        separators=[r"(?=Step \d+:)"],
        chunk_size=MAX_TOKEN_LENGTH * 3,
        chunk_overlap=0,
        is_separator_regex=True,
    ),
    "arguments": RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=MAX_TOKEN_LENGTH * 3,
        chunk_overlap=0,
    ),
    "intro": RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=MAX_TOKEN_LENGTH * 3,
        chunk_overlap=CHUNK_OVERLAP,
    ),
    "description": RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=MAX_TOKEN_LENGTH * 3,
        chunk_overlap=CHUNK_OVERLAP,
    )
}

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_doc(filepath: Path) -> tuple[list[Document], list[str]]:
    """
    Parse one Doxygen-generated code-module .md file.
    """

    documents: list[Document] = []
    ids: list[str] = []

    for section in h12_splitter.split_text(filepath.read_text(encoding="utf-8")):
        h1 = section.metadata.get("h1", "") #module name
        h2 = section.metadata.get("h2", "") #variable, subroutine::subroutine_name, function::function_name
        content = section.page_content.strip()

        section_id = "/".join(p for p in [h1, h2] if p)

        if "variable" in h2:
            # each variable is a document
            for ivar in content.splitlines():
                check_chunk_length(ivar.strip(), end_in_error=True)
                name = f"{section_id}/{ivar.strip()}"
                metadata = ChunkMetadata(
                    source=h1, 
                    name=name,
                    parent=h1, 
                    datatype="variable"
                )
                documents.append(Document(page_content=ivar.strip(), metadata=metadata.model_dump()))
                ids.append(name)

        elif "subroutine" in h2 or "function" in h2:
            # Procedure block — split by H3 sub-section
            for subsection in h3_splitter.split_text(content):
                h3 = subsection.metadata.get("h3")
                subsection_id = f"{section_id}/{h3}"
                splitter = SUBSECTION_SPLITTERS.get(h3)
                for ichunk, chunk in enumerate(splitter.split_text(subsection.page_content), start=1):
                    name = f"{subsection_id}/chunk{ichunk}"
                    metadata = ChunkMetadata(
                        source=h1,
                        name=name,
                        parent=section_id,
                        datatype="procedure",
                        ichunk=ichunk
                    )
                    documents.append(Document(page_content=chunk.strip(), metadata=metadata.model_dump()))
                    ids.append(name)

    return documents, ids

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(code_mods_dir: Path|str, create_database: bool = False) -> tuple[list[Document], list[str]] | None:
    """Parse all code-module .md files.
    
    Args:
        code_mods_dir: Directory containing the Doxygen-generated code-module markdown files.
        create_database: If True, ingest into FMSCouplerCode Milvus collection.
                        If False, return the documents and ids instead.
    
    Returns:
        If create_database is False: (all_documents, all_ids)
        If create_database is True: None
    """
    filepaths = [Path(code_mods_dir) / filename for filename in CODE_MODULE_FILES]
    for filepath in filepaths:
        if not filepath.exists():
            raise FileNotFoundError(f"  {filepath.name}  MISSING")

    all_documents: list[Document] = []
    all_ids: list[str] = []

    for filepath in filepaths:
        docs, ids = parse_doc(filepath)
        all_documents.extend(docs)
        all_ids.extend(ids)
        print(f"  {filepath.name:<45}  {len(docs):>3} documents")

    print(f"\nTotal: {len(all_documents)} documents")
    
    if create_database:
        create_milvus_database(all_documents, all_ids, COLLECTION_NAME)
    else:
        return all_documents, all_ids

# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------

TESTS = [
    ("What does coupler_init do and what arguments does it take?",       None),
    ("What arguments does flux_exchange_init accept?",                   None),
    ("How does sfc_boundary_layer work step by step?",                   None),
    ("What module variables are defined in full_coupler_mod?",           None),
    ("What are the steps in the atmos_ocean_fluxes_calc flowchart?",     None),
    ("How does land_ice_flux_exchange compute turbulent fluxes?",        None),
    ("What subroutines does atm_land_ice_flux_exchange_mod provide?",    None),
    ("How are ice-ocean fluxes calculated in ice_ocean_flux_exchange?",  None),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build(code_mods_dir=Path("."), create_database=True)
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
