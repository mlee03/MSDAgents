from pathlib import Path

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from shared.collection_data import CollectionData

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=True,
    return_each_line=False
)

chunker = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " ", ""],
    chunk_size=500,
    chunk_overlap=50
)

def construct_name(section):

    #get text under header
    h1 = section.metadata.get("h1", "").strip()
    h2 = section.metadata.get("h2", "").strip()
    h3 = section.metadata.get("h3", "").strip()
    return "/".join([h for h in (h1, h2, h3) if h])


def chunk(text: str) -> dict:
    splitted_content = chunker.split_text(text.strip())       
    is_chunked = False if len(splitted_content) == 1 else True
    chunks = list(range(1, len(splitted_content) + 1))
    return {
        "splitted_content": splitted_content,
        "is_chunked": is_chunked,
        "chunks": chunks
    }


def parse(mddir: Path|str, mdfile: Path|str):
    data = []
    mdfile_ = Path(mddir)/Path(mdfile)
    for section in header_splitter.split_text(mdfile_.read_text()):
        chunkdict = chunk(section.page_content)
        name = construct_name(section)
        for ichunk, chunk_text in enumerate(chunkdict["splitted_content"], start=1):
            datum = CollectionData(
                    sourcefile=mdfile_.name,
                    name=name,
                    is_chunked=chunkdict["is_chunked"],
                    ichunk=ichunk,
                    chunks=chunkdict["chunks"],
                    text = chunk_text.strip()
            )
            data.append(datum)
    return data