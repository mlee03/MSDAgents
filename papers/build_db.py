"""
Build (or rebuild) the persistent Chroma vector database from the PDFs in this directory.

Usage:
    python build_db.py           # build; skip if DB already exists
    python build_db.py --rebuild # force a full rebuild
"""

import argparse
import shutil
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

PAPERS_DIR = Path(__file__).parent
DB_DIR = PAPERS_DIR / "chroma_db"
COLLECTION = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()


def build(rebuild: bool = False) -> None:
    if DB_DIR.exists():
        if not rebuild:
            print(f"Database already exists at {DB_DIR}. Use --rebuild to overwrite.")
            return
        print(f"Removing existing database at {DB_DIR} …")
        shutil.rmtree(DB_DIR)

    print("Loading PDFs …")
    loader = PyPDFDirectoryLoader(str(PAPERS_DIR))
    docs = loader.load()
    print(f"  Loaded {len(docs)} pages from {len({d.metadata['source'] for d in docs})} PDFs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    print("Embedding and saving to Chroma …")
    embeddings = LocalEmbeddings()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    Chroma.from_documents(
        chunks,
        embeddings,
        client=client,
        collection_name=COLLECTION,
    )
    print(f"  Done. Database saved to {DB_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the DB")
    args = parser.parse_args()
    build(rebuild=args.rebuild)
