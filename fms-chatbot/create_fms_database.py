from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pymilvus import DataType, MilvusClient

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


def ensure_collection(client: MilvusClient, collection_name: str, recreate: bool) -> None:
    """Create collection if missing, or recreate if requested."""
    exists = client.has_collection(collection_name=collection_name)
    if exists and recreate:
        client.drop_collection(collection_name=collection_name)
        exists = False

    if not exists:
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="kind", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="xml_file", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=2)

        client.create_collection(collection_name=collection_name, schema=schema)


def batch_chunks(items: list[Document], chunk_size: int) -> Iterable[list[Document]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def write_markdown_export(documents: list[Document], markdown_path: Path) -> None:
    """Write parsed XML-derived documents to markdown for human review."""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    lines: list[str] = [
        "# FMS Parsed XML Export",
        "",
        f"Generated UTC: {timestamp}",
        f"Document count: {len(documents)}",
        "",
        "---",
    ]

    for idx, doc in enumerate(documents, start=1):
        metadata = dict(doc.metadata)
        lines.extend(
            [
                "",
                f"## Document {idx}",
                "",
                f"- name: {metadata.get('name', '')}",
                f"- kind: {metadata.get('kind', '')}",
                f"- source: {metadata.get('source', '')}",
                f"- xml_file: {metadata.get('xml_file', '')}",
                "",
                "### Text",
                "",
                "```text",
                doc.page_content,
                "```",
                "",
                "---",
            ]
        )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def ingest_documents(
    client: MilvusClient,
    collection_name: str,
    documents: list[Document],
    batch_size: int,
) -> int:
    """Insert parsed documents into Milvus."""
    inserted = 0
    default_embedding = [0.0, 0.0]

    for chunk in batch_chunks(documents, batch_size):
        rows: list[dict[str, str]] = []
        for doc in chunk:
            metadata = dict(doc.metadata)
            rows.append(
                {
                    "text": doc.page_content,
                    "source": str(metadata.get("source", "")),
                    "name": str(metadata.get("name", "")),
                    "kind": str(metadata.get("kind", "")),
                    "xml_file": str(metadata.get("xml_file", "")),
                    "embedding": default_embedding,
                }
            )

        client.insert(collection_name=collection_name, data=rows)

        inserted += len(chunk)

    return inserted


def connect_client(milvus_db_path: Path) -> MilvusClient:
    """Connect to a local Milvus Lite database file."""
    milvus_db_path.parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(uri=str(milvus_db_path))


def main() -> int:
    milvus_db_path = Path("/home/Ryan.Mulhall/msdagents/fms-chatbot/local_storage/fms_milvus.db")
    collection_name = "fms"
    xml_dir = "/home/Ryan.Mulhall/msdagents/fms/build_docs/docs/xml"
    markdown_export = "/home/Ryan.Mulhall/msdagents/fms-chatbot/parsed_fms_docs.md"
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

    write_markdown_export(documents=docs, markdown_path=Path(markdown_export))

    client = connect_client(milvus_db_path)
    try:
        ensure_collection(client, collection_name, recreate_collection)
        inserted = ingest_documents(client, collection_name, docs, batch_size)
    finally:
        if hasattr(client, "close"):
            client.close()

    print(f"Parsed documents: {len(docs)}")
    print(f"Markdown export: {markdown_export}")
    print(f"Inserted documents: {inserted}")
    print(f"Collection: {collection_name}")
    print(f"Milvus DB: {milvus_db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
