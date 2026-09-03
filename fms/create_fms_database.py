from pathlib import Path

from parsers.fortran_parser import doxygen_xml_parser
import parsers.markdownfile_parser as markdownfile_parser
import parsers.markdownfile_parser as markdownfile_parser

from shared.client import newCollection
from shared.utils import git_clone, run_doxygen

import fmsfiles

FMS_DIR = Path("./FMS")
DOXYGEN_DIR = FMS_DIR/"docs"
XML_DIR = DOXYGEN_DIR/"xml"
COLLECTION_NAME = "FMS"
MARKDOWN_DIR = FMS_DIR / "markdowns"

all_collection_data = []

for xmlfile in fmsfiles.FMS_GROUP_FILES:
    print(f"Processing XML file: {xmlfile}")
    modxml = doxygen_xml_parser.ModuleDocument(XML_DIR, xmlfile)
    modxml.populate()
    mdfile = modxml.write_markdown(output_dir=MARKDOWN_DIR)
    all_collection_data.extend(markdownfile_parser.parse(MARKDOWN_DIR, mdfile))

for mdfile in fmsfiles.FMS_MD_FILES:
    print(f"Processing Markdown file: {mdfile}")
    all_collection_data.extend(markdownfile_parser.parse(FMS_DIR, mdfile))

database = newCollection(COLLECTION_NAME, connect=True)
database.create_collection()
database.add_data(data=all_collection_data)
database.test_collection()

