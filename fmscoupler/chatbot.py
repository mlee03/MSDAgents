from shared.chatbot import RAGChatbot
from shared.client import Client, MilvusRetriever

OLLAMA_CHAT_MODEL = "mistral-nemo:latest"
COLLECTION_NAME = "FMSCoupler"

system_message = """
You are a technical assistant for the FMSCoupler.  

## Background
FMSCoupler, Flexible Modeling Systems Coupler, is a set of program
and modules to couple the atmosphere, ocean, land, and ice components in the 
GFDL (Geophysical Fluid Dynamics Laboratory) coupled climate models.

## Instructions:
- Use only the supplied context to answer.
- If the context does not contain the answer, say you do not know.
- Answer with a concise explanation.  Do not use markdown formatting.

## Context: 
{context}
"""

client = Client(COLLECTION_NAME)
retriever = MilvusRetriever(client)

chatbot = RAGChatbot(
    retriever=retriever, 
    system_message=system_message,
    model_name=OLLAMA_CHAT_MODEL, 
)

while True:
    user_question = input("\nYou: ").strip()
    if user_question.lower() in {"quit", "exit", "q"}:
        print("Bye.")
        break

    response, docs_and_scores, context = chatbot.ask(user_question)        
    print(f"\nAssistant: {response}")
    print(f"source: {[doc['sourcefile'] for doc in docs_and_scores]}")
    print("\n\n")
