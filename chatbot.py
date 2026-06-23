from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

OLLAMA_CHAT_MODEL = "mistral-nemo:latest"
HYBRID_LIMIT = 24

class RAGChatbot:

    def __init__(self, vectorstore: Any, temperature: float = 0, model_name: str = OLLAMA_CHAT_MODEL, search_hybrid = False):
        
        self.chatbot = ChatOllama(model=model_name, temperature=temperature)
        
        # Load unified vectorstore
        self.vectorstore = vectorstore

        # Set up hybrid search if requested
        self.search_hybrid = search_hybrid
        self.hybrid_kwargs = {"ranker_type": "rrf", "ranker_params": {"k": 60}} if search_hybrid else {}

        self.system = """
            ## Instructions:
            You are a technical assistant for the FMSCoupler codebase.
            FMSCoupler (Flexible Modeling Systems Coupler) is a program and 
            a set of modules used in climate modeling.  Use only the supplied context to answer.
            If the context does not contain the answer, say you do not know.
            Else, answer with a concise explanation.  Do not use markdown formatting.

            ## Context: 
            {context}
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.system), ("human", "{question}")]
        )

        self.answer_chain = self.prompt | self.chatbot | StrOutputParser()
        

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        """Search unified vectorstore and assemble sibling chunks by parent."""

        docs_and_scores = self.vectorstore.similarity_search_with_score(
            question, k=HYBRID_LIMIT, **self.hybrid_kwargs
        )

        # Deduplicate by parent
        #parents = list(set(doc.metadata.get("parent") for doc, _ in docs_and_scores))

        # Aggregate sibling chunks for each parent
        #for parent in parents:
        #    expr = f'parent == "{parent}"'
        #    siblings = self.vectorstore.col.query(expr=expr, output_fields=["text", "ichunk"])
        #    if siblings:
        #        siblings_sorted = sorted(siblings, key=lambda r: r.get("ichunk", 1))
        #        doc.page_content = "  ".join(r["text"] for r in siblings_sorted)

        return docs_and_scores

    def ask(self, question: str) -> tuple[str, list[tuple[Document, float]]]:
        """Invoke"""
        docs_and_scores = self.retrieve(question)
        context = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])
        answer = self.answer_chain.invoke({"question": question, "context": context})
        return answer, docs_and_scores
