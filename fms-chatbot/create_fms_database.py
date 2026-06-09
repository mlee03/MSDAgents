from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from langchain_core.documents import Document
from weaviate.classes.config import DataType, Property
import weaviate

from doxygen_xml_parser import ModuleBodyDocument


def parse_xml_directory(xml_dir: Path) -> list[Document]:
    """Parse Doxygen module XML files into LangChain documents."""
    docs: list[Document] = []

    xml_files = sorted(Path(xml_dir).glob("*.xml"))
    if not xml_files:
        return docs

    for xml_file in xml_files:
        xml_name = xml_file.name
        if not xml_name.startswith("namespace"):
            continue
        if not xml_name.endswith("__mod.xml"):
            continue

        try:
            parsed = _parse_single_module(xml_dir=xml_dir, xml_name=xml_name)
        except Exception as exc:
            print(f"Skipping {xml_name}: {exc}")
            continue
        docs.extend(parsed)

    return docs


def _parse_single_module(xml_dir: Path, xml_name: str) -> list[Document]:
    """Parse one module XML file and return procedure + variable documents."""
    try:
        module_doc = ModuleBodyDocument(
            xmldir=xml_dir,
            xmlfile=xml_name,
            append_overview=True,
        )
    except Exception:
        # Some module files may not have a matching top-level overview XML.
        module_doc = ModuleBodyDocument(
            xmldir=xml_dir,
            xmlfile=xml_name,
            append_overview=False,
        )

    module_doc.document_module_variables()
    module_doc.document_procedures()

    out_docs: list[Document] = []
    for _, doc in module_doc.variables.items():
        metadata = dict(doc.metadata)
        metadata["kind"] = "variable"
        metadata["xml_file"] = xml_name
        out_docs.append(Document(page_content=doc.page_content, metadata=metadata))

    for _, doc in module_doc.procedures.items():
        metadata = dict(doc.metadata)
        metadata["kind"] = "procedure"
        metadata["xml_file"] = xml_name
        out_docs.append(Document(page_content=doc.page_content, metadata=metadata))

    return out_docs


def ensure_collection(client: weaviate.WeaviateClient, collection_name: str, recreate: bool) -> None:
    """Create collection if missing, or recreate if requested."""
    exists = client.collections.exists(collection_name)
    if exists and recreate:
        client.collections.delete(collection_name)
        exists = False

    if not exists:
        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="name", data_type=DataType.TEXT),
                Property(name="kind", data_type=DataType.TEXT),
                Property(name="xml_file", data_type=DataType.TEXT),
            ],
        )


def batch_chunks(items: list[Document], chunk_size: int) -> Iterable[list[Document]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def ingest_documents(
    client: weaviate.WeaviateClient,
    collection_name: str,
    documents: list[Document],
    batch_size: int,
) -> int:
    """Insert parsed documents into Weaviate."""
    collection = client.collections.get(collection_name)
    inserted = 0

    for chunk in batch_chunks(documents, batch_size):
        with collection.batch.dynamic() as batch:
            for doc in chunk:
                metadata = dict(doc.metadata)
                batch.add_object(
                    properties={
                        "text": doc.page_content,
                        "source": metadata.get("source", ""),
                        "name": metadata.get("name", ""),
                        "kind": metadata.get("kind", ""),
                        "xml_file": metadata.get("xml_file", ""),
                    }
                )

        inserted += len(chunk)

    return inserted


def connect_client(weaviate_url: str) -> weaviate.WeaviateClient:
    """Connect to a local/self-hosted Weaviate instance via URL."""
    parsed = urlparse(weaviate_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    return weaviate.connect_to_local(host=host, port=port)


def main() -> int:
    weviate_url = "http://localhost:8080"
    collection_name = "fms"
    xml_dir = "/home/Ryan.Mulhall/msdagents/fms/build_docs/docs/xml"
    batch_size = 100
    recreate_collection = True
    if not Path(xml_dir).exists() or not Path(xml_dir).is_dir():
        Path(xml_dir).mkdir(parents=True, exist_ok=True)
    if batch_size <= 0:
        print("batch-size must be > 0")
        return 1

    docs = parse_xml_directory(xml_dir)
    if not docs:
        print("No parseable module XML files found (expected namespace*__mod.xml files).")
        return 1

    client = connect_client(weviate_url)
    try:
        ensure_collection(client, collection_name, recreate_collection)
        inserted = ingest_documents(client, collection_name, docs, batch_size)
    finally:
        client.close()

    print(f"Parsed documents: {len(docs)}")
    print(f"Inserted documents: {inserted}")
    print(f"Collection: {collection_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
