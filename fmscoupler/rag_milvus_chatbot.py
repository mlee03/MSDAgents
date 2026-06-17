from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import ChatOllama

from create_milvus_one_hybrid_collection_langchain_markdown import (
    COLLECTION_NAME,
    HUGGINGFACE_MODEL,
    MILVUS_HOST,
    MILVUS_PORT,
)

OLLAMA_CHAT_MODEL = "mistral-nemo:latest"
HYBRID_LIMIT = 24

class RAGChatbot:

    def __init__(self, model_name: str = OLLAMA_CHAT_MODEL):
        
        self.chatbot = ChatOllama(model=model_name, temperature=0)
        self.dense_ef = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL, model_kwargs={"device": "cpu"})

        self.vectorstore = Milvus(
            embedding_function=self.dense_ef,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            collection_name=COLLECTION_NAME,
        )
        
        self.system = """
            You are a technical assistant for the FMSCoupler codebase.
            FMSCoupler is a program and a set of modules used in climate modeling.
            Use only the supplied context to answer.
            If the context does not contain the answer, say you do not know.
            Else, answer with a concise explanation.  Do not use markdown formatting.
            Context: 
            {context}
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.system), ("human", "{question}")]
        )

        self.answer_chain = self.prompt | self.chatbot | StrOutputParser()

    def retrieve_hybrid(self, question: str) -> list[tuple[Document, float]]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            question, k=HYBRID_LIMIT, ranker_type="rrf", ranker_params={"k": 60}
        )
        seen_parents: set[str] = set()
        parsed_doc = []
        for doc, score in docs_and_scores:
            parent = doc.metadata.get("parent")
            if parent not in seen_parents:
                seen_parents.add(parent)
                parsed_doc.append((doc, score))
        docs_and_scores = parsed_doc

        for doc, score in docs_and_scores:
            parent = doc.metadata.get("parent")
            expr = f'parent == "{parent}"'
            siblings = self.vectorstore.col.query(expr=expr, output_fields=["text", "ichunk"])
            siblings_sorted = sorted(siblings, key=lambda r: r.get("ichunk", 1))
            doc.page_content = "  ".join(r["text"] for r in siblings_sorted)

        return docs_and_scores

    def ask(self, question: str) -> tuple[str, list[tuple[Document, float]]]:
        docs_and_scores = self.retrieve_hybrid(question) #returns [(Document, score), ...]
        context = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])
        print(context)
        answer = self.answer_chain.invoke({"question": question, "context": context})
        return answer, docs_and_scores


if __name__ == "__main__":
    chatbot = RAGChatbot()

    while True:
        user_question = input("\nYou: ").strip()
        if user_question.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        response, docs_and_scores = chatbot.ask(user_question)        
        sources = ", ".join([doc.metadata.get("source")+"/"+doc.metadata.get("name") for doc, _ in docs_and_scores])
        print(f"\nAssistant: {response}")
        print(f"Relevant sources: {sources}")
        print("\n\n")

