"""
RAG chatbot over the PDF papers in this directory.

Usage:
    python rag_chatbot.py            # build DB then chat
    python rag_chatbot.py --rebuild  # force rebuild the vector DB
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

PAPERS_DIR = Path(__file__).parent
DB_DIR = PAPERS_DIR / "chroma_db"
COLLECTION = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()


def open_vectorstore(embeddings: Embeddings) -> Chroma:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return Chroma(client=client, collection_name=COLLECTION, embedding_function=embeddings)


def build_chain(db: Chroma) -> object:
    retriever = db.as_retriever(search_kwargs={"k": 6})
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    system_prompt = (
        "You are a helpful research assistant with expertise in climate modeling. "
        "Answer the user's question using ONLY the context excerpts provided below. "
        "If the answer is not contained in the context, say so clearly.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve docs, format them, run LLM, return answer + raw docs
    answer_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(answer=answer_chain, context=retriever)


def chat_loop(chain) -> None:
    print("\nRAG Chatbot ready. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        result = chain.invoke(question)
        print(f"\nAssistant: {result['answer']}\n")

        sources = {Path(d.metadata["source"]).name for d in result.get("context", [])}
        if sources:
            print("Sources:", ", ".join(sorted(sources)), "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()  # --help support

    if not DB_DIR.exists():
        raise SystemExit(f"Database not found at {DB_DIR}. Run build_db.py first.")

    print(f"Loading vector store from {DB_DIR} …")
    embeddings = LocalEmbeddings()
    db = open_vectorstore(embeddings)

    chain = build_chain(db)
    chat_loop(chain)


if __name__ == "__main__":
    main()
