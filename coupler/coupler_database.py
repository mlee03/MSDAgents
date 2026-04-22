from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup


class XMLsoup():

    def __init__(self, codebase: str, xmldir: str|Path = "./", xmlfile: str|Path = "None"):

        self.codebase = codebase
        self.xmldir = Path(xmldir)
        self.xmlfile = Path(xmlfile)
        self.soup = self.get_xmlsoup()
        self.toplevel_name = self.get_name(toplevel=True)
        
    def get_xmlsoup(self):

        xmlfile = self.xmldir/self.xmlfile

        if xmlfile.exists():
            with open(xmlfile, "r") as openedfile:
                return BeautifulSoup(openedfile, "lxml-xml")
        else:
            raise IOERrror("xml file '{xmlfile}' in directory '{xmldir}' does not exist")        

        
    def get_name(self, soup = None, toplevel = False):
        
        if soup is None:
            soup = self.soup

        tag = "name"
        if toplevel:
            tag = "compoundname"

        name = self.get_tag(tag, soup)

        if name is None:
            raise RuntimeError(f"cannot find tag 'compoundname' to set name in {self.soup}")
                
        print(name)
        return name

    
    def get_tag(self, tag, soup = None):

        if soup is None:
            soup = self.soup

        tagobj = soup.find(tag)                
        
        if tagobj is not None:
            tagstr = tagobj.text.strip()
            if tagstr:
                return tagstr
            
        return None


    def get_parameters_description(self, soup = None):

        if soup is None:
            soup = self.soup
                
        parameter_item_objs = soup.find_all("parameteritem")
        
        if parameter_item_objs is None:
            return None

        parameters_description = ""
        for parameter_item_obj in parameter_item_objs:
            description = self.get_parameter_description(parameter_item_obj)
            if description is not None:
                parameters_description += description
            
        return parameters_description
    
        
    def get_parameter_description(self, parameter_item_obj):

        parameter_namelist_obj = parameter_item_obj.parameternamelist
        if parameter_namelist_obj is None:
            return None

        inout = parameter_namelist_obj.get("direction")
        if inout is None: inout = "inout"
        
        parameter_name = parameter_namelist_obj.text.split()

        description = f"{parameter_name} is an intent({inout}) variable."
        
        parameter_description = self.get_tag("parameterdescription", parameter_item_obj)
        if parameter_description is not None:
            description += f"{parameter_name} {parameter_description}."

        return description


    
class f90XMLsoup(XMLsoup):

    def __init__(self,
                 codebase: str,
                 prog_or_mod: Literal["module", "program"] = "module",
                 xmldir: str|Path = "./docs/xml",
                 xmlfile: str|Path = None):

        super().__init__(codebase, xmldir, xmlfile)
        self.prog_or_mod = prog_or_mod
        self.overview = self.document_overview()

        
    def document_overview(self):
        
        briefdescription = self.get_tag("briefdescription")
        detaileddescription = self.get_tag("parblock")
        
        overview = f"{self.toplevel_name} is a {self.prog_or_mod} in {self.codebase}."
        
        if briefdescription is not None:
            overview += briefdescription

        if detaileddescription is not None:
            overview += detaileddescription

        print(overview)
        return overview
                    
    
class namespaceXMLsoup(XMLsoup):

    def __init__(self,
                 codebase: str,
                 prog_or_mod: Literal["module", "program"] = "module",                                                                               
                 xmldir: str|Path = "./docs/xml",
                 xmlfile: str|Path = None):
        super().__init__(codebase, xmldir, xmlfile)

        self.prog_or_mod = prog_or_mod
        self.documents = {
            "variables": {},
            "procedures": {}
        }

        
    def document_variables(self):

        variables_obj = self.soup.find_all("memberdef", {"kind": "variable"})

        if variables_obj is None:
            return
        
        documents = {}        
        for variable in variables_obj:
            
            varname = self.get_name(variable)
            vartype = self.get_tag("type", variable)
            briefdescription = self.get_tag("briefdescription", variable)

            var_description = f"{varname} is a {self.prog_or_mod} variable in {self.toplevel_name}."

            if vartype is not None:
                var_description += f"{varname} is a {vartype}."

            if briefdescription is not None:
                var_description += f"{varname} {briefdescription}."
            
            documents[varname] = {
                "document": var_description,
                "metadata": {"source": self.toplevel_name, "name": varname, "identity": "variable"},
                "id": varname
            }

        self.documents["variables"] = documents
            
            
    def document_procedures(self):

        procedures_obj = self.soup.find_all("memberdef", {"kind": "function"})
        if procedures_obj is None: return

        documents = {}
        for procedure in procedures_obj:
            
            procname = self.get_name(procedure)
            proctype = self.get_tag("type", procedure)
            argsstring = self.get_tag("argsstring", procedure)
            parameters_description = self.get_parameters_description(procedure)
            briefdescription = self.get_tag("briefdescription", procedure)
            detaileddescription = self.get_tag("parblock", procedure)            
            inbodydescription = self.get_tag("inbodydescription", procedure)
            
            if proctype is None:
                proctype = "procedure"

            procedure_description= f"{procname} is a {proctype} in {self.toplevel_name}."
            
            if argsstring is None:
                procedure_description += f"{procname} does not have any {proctype} arguments."
            else:
                procedure_description += f"{procname} has these {proctype} arguments:  {argsstring}."

            if parameters_description is not None:
                procedure_description += parameters_description

            if briefdescription is not None:
                procedure_description += briefdescription

            if detaileddescription is not None:
                procedure_description += detaileddescription

            if inbodydescription is not None:
                procedure_description += f"Inside {procname}, the following occurs: {inbodydescription}"
                
            documents[procname] = {
                "document": procedure_description,
                "metadata": {"source": self.toplevel_name, "name": procname, "identity": proctype},
                "id": procname
            }

            print(documents)
            exit()
        self.documents["procedures"] = documents
            
                    
modxml = namespaceXMLsoup(codebase="FMSCoupler", xmlfile="namespaceatm__land__ice__flux__exchange__mod.xml")
modxml.document_variables()
modxml.document_procedures()
print(modxml.documents["variables"])
print(modxml.documents["procedures"])
