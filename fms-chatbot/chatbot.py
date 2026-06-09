from __future__ import annotations

from typing import Any

import weaviate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


LLM_MODEL = "llama3.2"
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080
COLLECTION_NAME = "fms"
TOP_K = 5


def fetch_context(collection: Any, query: str, limit: int = TOP_K) -> str:
  """Retrieve top matching docs from Weaviate and format them as context."""
  result = collection.query.bm25(
    query=query,
    query_properties=["text", "name", "source"],
    limit=limit,
    return_properties=["text", "name", "source", "kind", "xml_file"],
  )

  chunks: list[str] = []
  for idx, obj in enumerate(result.objects, start=1):
    props = obj.properties or {}
    chunks.append(
      (
        f"Context {idx}:\n"
        f"name: {props.get('name', '')}\n"
        f"kind: {props.get('kind', '')}\n"
        f"source: {props.get('source', '')}\n"
        f"xml_file: {props.get('xml_file', '')}\n"
        f"text: {props.get('text', '')}"
      )
    )

  if not chunks:
    return "No relevant context was found in the FMS Weaviate collection."

  return "\n\n".join(chunks)


def main() -> None:
  chatbot = ChatOllama(model=LLM_MODEL, microstat_tau=2.0)

  client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
  try:
    if not client.collections.exists(COLLECTION_NAME):
      raise RuntimeError(
        f"Collection '{COLLECTION_NAME}' not found. Run create_fms_database.py first."
      )

    collection = client.collections.get(COLLECTION_NAME)

    system_message = SystemMessage(
      content=(
        "FMS is a Fortran library used for scientific computing. You are an FMS coding assistant. "
        "Only answer questions using the retrieved context. "
        "If context is insufficient, say you do not have enough information "
        "from the indexed FMS docs."
      )
    )

    intro_query = "Introduce yourself. Ask how may I assist you?"
    intro_context = fetch_context(collection, intro_query)
    intro = chatbot.invoke(
      [
        system_message,
        HumanMessage(
          content=(
            f"Question: {intro_query}\n\n"
            f"Retrieved context:\n{intro_context}\n\n"
            "Answer using only the retrieved context."
          )
        ),
      ]
    )

    print(intro.content)
    print(">", end=" ")

    while True:
      query = input().strip()
      if not query:
        print(">", end=" ")
        continue
      if "goodbye" in query.lower() or query.lower() in {"exit", "quit"}:
        print("Goodbye.")
        break

      context = fetch_context(collection, query)
      answer = chatbot.invoke(
        [
          system_message,
          HumanMessage(
            content=(
              f"Question: {query}\n\n"
              f"Retrieved context:\n{context}\n\n"
              "Answer using only the retrieved context."
            )
          ),
        ]
      )
      print(answer.content)
      print(">", end=" ")
  finally:
    client.close()


if __name__ == "__main__":
  main()
               
