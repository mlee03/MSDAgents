import chromadb

from coupler_database import (
    f90XMLsoup,
    namespaceXMLsoup
)

#THIS IS HORRIBLE AND WILL CHANGE TO BE SIMPLER

codebase = "fmscoupler"
xmldir = "./docs/xml"

class Collection():
    def __init__(self, namespace_xmlfile = None, f90_xmlfile = None, prog_or_mod = "module", name="None"):

        self.namespace_xmlfile = namespace_xmlfile
        self.f90_xmlfile = f90_xmlfile
        self.prog_or_mod = prog_or_mod
        self.name = name

        self.namespacesoup = None
        if self.namespace_xmlfile is not None:
            self.namespacesoup = namespaceXMLsoup(
                codebase,
                prog_or_mod,
                xmldir=xmldir,
                xmlfile=self.namespace_xmlfile
            )

        self.f90soup = None
        if self.f90_xmlfile is not None:
            self.f90soup = f90XMLsoup(
                codebase,
                prog_or_mod,
                xmldir=xmldir,
                xmlfile=self.f90_xmlfile
            )

collections = [
    Collection(        
        namespace_xmlfile="namespaceatm__land__ice__flux__exchange__mod.xml",
        f90_xmlfile = "atm__land__ice__flux__exchange_8_f90.xml",
        prog_or_mod = "module",
        name = "atm_land_ice_flux_exchange_mod"
    ),
    Collection(
        namespace_xmlfile = None, 
        f90_xmlfile = "full_2coupler__main_8_f90.xml",
        prog_or_mod = "program",
        name = "coupler_main"
    )
]

def make_chroma_db():

    client = chromadb.PersistentClient("./coupler-chromadb")

    
    for collection in collections:

        chromadb_collection = client.get_or_create_collection(name=collection.name) #metadata={}

        if collection.f90soup is not None:

            collection.f90soup.document_overview()
            print(collection.f90soup.overview)
            chromadb_collection.upsert(ids=["overview"], documents=[collection.f90soup.overview])
            
        if collection.namespacesoup is not None:

            collection.namespacesoup.document_variables()
            collection.namespacesoup.document_procedures()
            doc_dict = collection.namespacesoup.documents
            
            ids, metadatas, documents = [], [], []
            for (key, thislist) in [("id", ids), ("metadata", metadatas), ("document", documents)]:
                thislist += [vardict[key] for var, vardict in doc_dict["variables"].items()]
                thislist += [procdict[key] for proc, procdict in doc_dict["procedures"].items()]

            chromadb_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    
make_chroma_db()
