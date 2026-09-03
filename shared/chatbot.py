from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

OLLAMA_CHAT_MODEL = "mistral-nemo:latest"
HYBRID_LIMIT = 24

class RAGChatbot:

    def __init__(self,
                 retriever,
                 system_message: str, 
                 temperature: float = 0,
                 model_name: str = OLLAMA_CHAT_MODEL,
                 search_hybrid = False,
    ):
        
        self.chatbot = ChatOllama(model=model_name, temperature=temperature)
        
        # Load vectorstore
        self.retriever = retriever
            
        self.system_message = system_message

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_message), ("human", "{question}")]
        )

        self.answer_chain = self.prompt | self.chatbot | StrOutputParser()

    
    def ask(self, question: str):
        """Invoke"""
        retrieved_data = self.retriever.retrieve(question)
        context = "\n\n".join([data["text"] for data in retrieved_data])
        answer = self.answer_chain.invoke({"question": question, "context": context})
        return answer, retrieved_data, context
