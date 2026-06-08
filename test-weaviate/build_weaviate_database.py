import weaviate
from weaviate.classes.config import Configure, Property, DataType
from pathlib import Path

import doxygen_xml_parser

modulefiles = {
    "atm_land_ice_flux_exchange_mod": {
        "toplevel": "atm__land__ice__flux__exchange_8_f90.xml",
        "body": "namespaceatm__land__ice__flux__exchange__mod.xml"
    },
    "atmos_ocean_dep_fluxes_calc_mod": {
        "toplevel": "atmos__ocean__dep__fluxes__calc_8_f90.xml",
        "body": "namespaceatmos__ocean__dep__fluxes__calc__mod.xml"
    },
    "atmos_ocean_fluxes_calc_mod": {
        "toplevel": "atmos__ocean__fluxes__calc_8_f90.xml",
        "body": "namespaceatmos__ocean__fluxes__calc__mod.xml"
    },
    "flux_exchange_mod": {
        "toplevel": "full_2flux__exchange_8_f90.xml",
        "body": "namespaceflux__exchange__mod.xml"
    },
    "full_coupler_mod": {
        "toplevel": "full__coupler__mod_8_f90.xml",
        "body": "namespacefull__coupler__mod.xml"
    },
    "ice_ocean_flux_exchange_mod": {
        "toplevel": "ice__ocean__flux__exchange_8_f90.xml",
        "body": "namespaceice__ocean__flux__exchange__mod.xml"
    },
    "land_ice_flux_exchange_mod": {
        "toplevel": "land__ice__flux__exchange_8_f90.xml",
        "body": "namespaceland__ice__flux__exchange__mod.xml"
    }
}

READMES = [
    "full/docs/README.md",
    "full/docs/FLUX.md",
    "full/docs/AtmosDataType.md",
    "full/docs/AtmosIceBoundaryType.md",
    "full/docs/AtmosLandBoundaryType.md",
    "full/docs/IceDataType.md",
    "full/docs/IceOceanBoundaryType.md",
    "full/docs/IceOceanDriverType.md",
    "full/docs/LandDataType.md",
    "full/docs/LandIceAtmosBoundaryType.md",
    "full/docs/OceanIceBoundaryType.md",
    "full/docs/OceanPublicType.md",
    "full/docs/OceanStateType.md"
]

#client = weaviate.connect_to_local(
#    host="localhost",
#    port=8081,
#    grpc_port=50051
#)

with weaviate.connect_to_local() as client:
    
    COLLECTION_NAME = "FMSCoupler"
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)    

    collection = client.collections.create(
        name=COLLECTION_NAME,
        vector_config=Configure.Vectors.text2vec_ollama(
            api_endpoint='http://localhost:11434',
            model="nomic-embed-text"
        ),
        properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="type", data_type=DataType.TEXT),
                Property(name="description", data_type=DataType.TEXT)
            ]
        )

    # parse doxygen xml files
    for module, files in modulefiles.items():
        toplevel = files["toplevel"]
        body = files["body"]

        body_doc = doxygen_xml_parser.ModuleBodyDocument(codebase="FMSCoupler", xmlfile=body)
        body_doc.document_module_variables()
        body_doc.document_procedures()

        toplevel_doc = doxygen_xml_parser.ModuleTopLevelDocument(codebase="FMSCoupler", xmlfile=toplevel)    

        collection.data.insert(toplevel_doc.overview)
        for varname, vardoc in body_doc.variable_docs.items():
            collection.data.insert(vardoc)
        for procname, procdoc in body_doc.procedures_doc.items():
            collection.data.insert(procdoc)


    