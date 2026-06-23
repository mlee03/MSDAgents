"""
create_unified_database.py

Unified database builder: combines narrative documentation and Doxygen-generated
code documentation into a single FMSCoupler collection.

  FMSCouplerDocs  — narrative documentation  (create_readme_collection.py)
  FMSCouplerCode  — Doxygen module/subroutine docs (create_code_module_collection.py)
  FMSCoupler      — unified collection combining both sources

Usage
-----
    python create_unified_database.py   # build unified collection
"""

from pathlib import Path

import create_readme_collection
import create_code_module_collection
from create_database_utils import create_milvus_database, test_collection

UNIFIED_COLLECTION_NAME = "FMSCoupler"
UNIFIED_LOG_FILE = "create_unified_database.log"
DOCS_DIR = Path("full/docs")
CODE_MODS_DIR = Path(".")

UNIFIED_TESTS = [
    # Documentation tests
    ("What are the model component state types in the full coupler?",      None),
    ("How does the fast loop implicit tridiagonal diffusion scheme work?", None),
    ("What are the fields in atmos_data_type for radiative fluxes?",      "datatype"),
    ("What fields does ice_ocean_boundary_type pass from ice to ocean?",  "boundary_type"),
    ("How is MPI PE layout configured for atmosphere and ocean?",          "overview"),
    
    # Code tests
    ("What does coupler_init do and what arguments does it take?",       None),
    ("How does sfc_boundary_layer work step by step?",                   None),
    ("What module variables are defined in full_coupler_mod?",           None),
    ("What are the steps in the atmos_ocean_fluxes_calc flowchart?",     None),
    ("How does land_ice_flux_exchange compute turbulent fluxes?",        None),
]

if __name__ == "__main__":
    print("=" * 72)
    print("Building unified FMSCoupler collection")
    print("=" * 72)
    
    # Get documents from both sources without creating separate databases
    print("parse Gathering narrative documentation...")
    readme_docs, readme_ids = create_readme_collection.build(docs_dir=DOCS_DIR, create_database=False)
    print(f"  -> {len(readme_docs)} documents from narrative docs")
    
    print("parse Gathering code module documentation...")
    code_docs, code_ids = create_code_module_collection.build(code_mods_dir=CODE_MODS_DIR, create_database=False)
    print(f"  -> {len(code_docs)} documents from code modules")
    
    # Combine all documents and ids
    all_documents = readme_docs + code_docs
    all_ids = readme_ids + code_ids
    
    print(f"\n[chunk] Total combined: {len(all_documents)} documents")
    print(f"[build] Creating unified Milvus collection: {UNIFIED_COLLECTION_NAME}")
    
    # Create the unified database
    create_milvus_database(all_documents, all_ids, UNIFIED_COLLECTION_NAME)
    
    # Test the unified collection
    print(f"\n[test] Testing unified collection...")
    test_collection(UNIFIED_COLLECTION_NAME, UNIFIED_TESTS, UNIFIED_LOG_FILE)
    
    print(f"\nDone. Unified collection available:")
    print(f"  {UNIFIED_COLLECTION_NAME:<24}  log: {UNIFIED_LOG_FILE}")
