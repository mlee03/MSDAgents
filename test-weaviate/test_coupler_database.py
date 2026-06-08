import weaviate
from langchain_ollama import ChatOllama 

PRIMARY_COLLECTION = "FMSCoupler"
README_COLLECTION = "FMSCouplerReadmes"

limit = 5
query = "What is FMSCoupler?"

def print_pretty(result) -> None:
    for i, obj in enumerate(result.objects, start=1):
        props = obj.properties or {}
        print(f"[{i}]")
        print(f"doc_id: {props.get('doc_id', 'N/A')}")
        print(f"source: {props.get('source', 'N/A')}")
        print(f"type: {props.get('type', 'N/A')}")
        print(f"description: {props.get('description', 'N/A')}")
        print()

def format

with weaviate.connect_to_local() as client:
    if not client.collections.exists(PRIMARY_COLLECTION):
        raise RuntimeError(
            f"Collection '{PRIMARY_COLLECTION}' does not exist. "
            "Run build_coupler_database.py first."
        )

    primary_collection = client.collections.get(PRIMARY_COLLECTION)

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        result = collection.query.near_text(
            query=query,
            limit=limit,
            return_properties=["doc_id", "source", "type", "description"]
        )


        print("\nNEAR TEXT: FMSCoupler")
        search_collection(
            primary_collection,
            query,
            ["doc_id", "source", "type", "description"],
        )

        print("NEAR TEXT: FMSCouplerReadmes")
        search_collection(
            readme_collection,
            query,
            ["doc_id", "file_name", "content"],
        )




