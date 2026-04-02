"""
chat.py
=======

A command-line RAG (Retrieval-Augmented Generation) chatbot for querying source code 
and documentation. This version is optimized for the 2026 LangChain v1.2 ecosystem.

Key Features:
-------------
- Model: Uses Ollama (gemma3) for local inference via the `ollama` Python SDK.
- Embeddings: Uses `langchain-huggingface` to run local transformer models.
- Vector Store: Uses FAISS for high-performance local similarity search.
- Modular: Implements the 2026 standard 'langchain_text_splitters' and partner packages.

Requirements:
-------------
- Python 3.11+ (Recommended)
- Ollama server running (default: http://localhost:11434)
- Packages: ollama, langchain, langchain-huggingface, langchain-text-splitters, 
            langchain-community, faiss-cpu, sentence-transformers, unstructured[md].
"""

import os
import ollama
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
# Standard 2026 LangChain partner package imports
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
import logging
from typing import List, Dict, Any, Optional

import textwrap
import asyncio
import inspect

# --- Configuration & Paths ---
DATA_PATH = "fre-cli"
INDEX_PATH = "faiss_index"

# 2026 Defaults: gemma3 is the preferred lightweight model for coding tasks
OLLAMA_MODEL = "gemma3"
OLLAMA_URL: Optional[str] = None
SHOW_SOURCES = True
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Mapping file extensions to specific 2026 document loaders
EXTENSION_LOADERS: Dict[str, Any] = {
    ".md": UnstructuredMarkdownLoader,  # Uses 'unstructured' for clean markdown parsing
    ".py": TextLoader,                  # Treats Python files as raw text
    ".rst": TextLoader,                 # reStructuredText fallback to TextLoader
}

def load_documents() -> List[Any]:
    """
    Crawls the DATA_PATH and loads supported files using multithreaded loaders.
    
    Returns:
        List[Document]: A list of LangChain Document objects with source metadata.
    """
    logging.info(f"📂 Scanning {DATA_PATH} for .md, .py, and .rst documents...")
    files_to_process = []
    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext in EXTENSION_LOADERS:
                files_to_process.append(os.path.join(root, filename))

    def load_file(full_path: str) -> List[Any]:
        ext = os.path.splitext(full_path)[1]
        try:
            loader = EXTENSION_LOADERS[ext](full_path)
            docs = loader.load()
            for doc in docs:
                # Critical for RAG: Keep track of exactly where this information came from
                doc.metadata["source"] = full_path
            return docs
        except Exception as e:
            logging.warning(f"⚠️ Skipping {full_path}: {e}")
            return []

    # Parallel loading significantly speeds up indexing for large repos
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_file, files_to_process), total=len(files_to_process), desc="🔍 Loading files"))
        documents = [doc for sublist in results for doc in sublist]

    return documents


def create_or_load_index(documents: List[Any], chunk_size: int = 500, chunk_overlap: int = 150) -> Any:
    """
    Manages the FAISS vector store. Loads from disk if available, otherwise builds from scratch.
    
    In 2026, we use 'langchain_huggingface' for local embeddings to avoid cloud API costs.
    """
    logging.info("📦 Initializing embeddings (HuggingFace/sentence-transformers)...")

    # 'all-mpnet-base-v2' is a high-accuracy 2026 standard for technical documentation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Step 1: Check for existing serialized FAISS index
    if os.path.exists(INDEX_PATH):
        try:
            logging.info("Found existing FAISS index on disk; loading...")
            # allow_dangerous_deserialization is required for local FAISS loading in newer versions
            vs = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            logging.info("Loaded existing FAISS index.")
            return vs
        except Exception as e:
            logging.info(f"Could not load index ({e}); rebuilding...")

    # Step 2: Chunking logic (Now using the standalone langchain_text_splitters package)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logging.info("Splitting documents into chunks for embedding...")
    try:
        chunks = splitter.split_documents(documents)
    except Exception:
        chunks = documents

    # Step 3: Vectorization (This is the heavy compute part)
    logging.info(f"Embedding and adding {len(chunks)} chunks to FAISS...")
    try:
        vs = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to build FAISS index: {e}")

    # Step 4: Persist for future use
    try:
        vs.save_local(INDEX_PATH)
    except Exception:
        logging.info("Persistence failed; index will remain in-memory for this session.")

    return vs


def call_ollama(prompt: str, model: str = "gemma3", ollama_url: Optional[str] = None) -> str:
    """
    Communicates with the local Ollama server using the official Python SDK (v0.6+).
    Handles various response shapes returned by different Ollama versions.
    """
    # Attempting to use the Class-based Client interface
    if hasattr(ollama, "Client"):
        client = ollama.Client(host=ollama_url) if ollama_url else ollama.Client()
        try:
            resp = client.generate(model=model, prompt=prompt, options={"temperature": 0.2})
            return resp.get('response', str(resp))
        except Exception as e:
            logging.debug(f"Client-based call failed: {e}. Falling back to module-level call.")

    # Fallback to module-level functional API
    if hasattr(ollama, "generate"):
        resp = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.2})
        # Standard SDK response is a dict with a 'response' key
        if isinstance(resp, dict):
            return resp.get("response") or resp.get("text") or str(resp)
        return str(resp)

    raise RuntimeError("Ollama SDK (ollama-python) not found or API incompatible.")


def build_prompt_from_docs(query: str, docs: List[Any], max_chars_per_doc: int = 1500) -> str:
    """
    Constructs a RAG prompt that strictly limits the LLM to the provided context.
    """
    parts = [
        "You are an assistant answering questions about the fre-cli repository. "
        "Use ONLY the provided context to answer. Include filenames when referencing code.",
        "If the answer is not in the context, state that you do not know. Do NOT hallucinate.",
        "---",
        "Context from repository files:",
    ]
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        # Clean newlines for better prompt readability
        content = doc.page_content.strip().replace("\\n", "\n")
        snippet = content[:max_chars_per_doc]
        parts.append(f"Source {i} [{src}]:\n{snippet}\n---")

    parts.append(f"Question: {query}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def smart_print(text, width=100):
    """
    Custom console printer that wraps paragraphs while preserving list formatting.
    """
    for line in text.splitlines():
        if line.strip().startswith(("-", "*", "•", "1.", "2.")):
            print(line)
        else:
            print(textwrap.fill(line, width=width))


def main() -> None:
    """
    Main execution loop: Loads data, initializes RAG, and starts the chat interface.
    """
    ollama_model = OLLAMA_MODEL
    # Priority: Environment Variable > Code Constant
    ollama_url = os.environ.get("OLLAMA_URL") or OLLAMA_URL

    try:
        # Load local source files
        documents = load_documents()
        if not documents:
            logging.error(f"❌ No documents found in {DATA_PATH}. Check your directory structure.")
            return

        # Setup the vector store
        vectorstore = create_or_load_index(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        # We retrieve the top 6 most relevant snippets to provide enough context for complex code questions
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        logging.info(f"🧠 Bot ready using Ollama model: {ollama_model}")

        print("\n🤖 Assistant: Ready to chat about fre-cli! Type 'exit' to quit.\n")
        while True:
            try:
                query = input("🗨️  You: ")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Exiting chat.")
                break
            
            if query.lower() in ("exit", "quit"):
                break

            try:
                # RAG Workflow: 1. Retrieve -> 2. Augment -> 3. Generate
                docs = retriever.invoke(query) # Using .invoke() which is the 2026 standard over .get_relevant_documents()
                prompt = build_prompt_from_docs(query, docs)
                
                resp_text = call_ollama(prompt, model=ollama_model, ollama_url=ollama_url)

                print("\n🤖 Bot:")
                smart_print(resp_text)
                print()

                if SHOW_SOURCES:
                    print("📚 Sources used:")
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "unknown")
                        snippet = doc.page_content.strip().replace("\n", " ")
                        print(f"  {i}. 📄 {source} | {textwrap.shorten(snippet, width=120, placeholder='...')}")
                    print()
            except Exception as e:
                logging.error(f"❌ QA Error: {e}")
                continue
                
    except Exception as e:
        logging.error(f"❌ Initialization Error: {e}")


if __name__ == "__main__":
    main()