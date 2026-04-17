from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup


class XMLsoup():

    def __init__(self, codebase: str, xmldir: str|Path = "./", xmlfile: str|Path = "None"):

        self.codebase = codebase
        self.xmldir = Path(xmldir)
        self.xmlfile = Path(xmlfile)
        self.soup = self.get_xmlsoup()
        self.modname = self.get_modname()
        
    def get_xmlsoup(self):

        xmlfile = self.xmldir/self.xmlfile
        if xmlfile.exists():
            with open(xmlfile, "r") as openedfile:
                return BeautifulSoup(openedfile, "lxml-xml")
        else:
            raise IOERrror("xml file '{xmlfile}' in directory '{xmldir}' does not exist")        
                
    def get_modname(self):
        modname_obj = self.soup.find("compoundname")
        if modname_obj is None:
            raise RuntimeError(f"cannot find tag 'compoundname' to set name in \n{self.soup}")
        return modname_obj.text.strip()

    
class f90XMLsoup(XMLsoup):

    def __init__(self,
                 codebase: str,
                 prog_or_mod: Literal["module", "program"] = "module",
                 xmldir: str|Path = "./docs/xml",
                 xmlfile: str|Path = None):
        super().__init__(codebase, xmldir, xmlfile)
        self.prog_or_mod = prog_or_mod
        self.documents = None
        self.metadatas = None
        self.ids = None
        self.description = self.get_toplevel_doc()

    def get_introduction(self):
        self.introduction = xml.soup.parblock.text.strip()

    def get_toplevel_doc(self):
        
        briefdescription_obj = self.soup.briefdescription
        detaileddescription_obj = self.soup.detaileddescription

        self.description = f"{self.modname} is a {self.prog_or_mod} in {self.codebase}"
        
        if briefdescription_obj is not None:
            briefdescription = briefdescription_obj.text.strip()
            if briefdescription:
                self.description += briefdescription

        if detaileddescription_obj is not None:
            parblock_obj = detaileddescription_obj.parblock
            if parblock_obj is not None:
                detaileddescription = detaileddescription_obj.text.strip()
                if detaileddescription:
                    self.description += detaileddescription


class namespaceXMLsoup(XMLsoup):

    def __init__(self,
                 codebase: str,
                 prog_or_mod: Literal["module", "program"] = "module",                                                                               
                 xmldir: str|Path = "./docs/xml",
                 xmlfile: str|Path = None):
        super().__init__(codebase, xmldir, xmlfile)

        self.prog_or_mod = prog_or_mod
        self.vardocs = {}
        self.procdocs = {}

        
    def set_variable_docs(self):

        variables_obj = self.soup.find_all("memberdef", {"kind": "variable"})
        if variables_obj is None: return

        for variable in variables_obj:
            
            varname = self.get_name(variable)
            vartype = self.get_type(variable)
            var_introsentence = f"{varname} is a {self.prog_or_mod} variable of type {vartype} in {self.modname}."

            vardef_sentence = self.get_briefdescription_as_sentence(variable, varname)
            
            self.vardocs[varname] = {
                "document": f"{var_introsentence} {vardef_sentence}",
                "metadata": {"source": self.modname, "name": varname, "identity": "variable"}
            }
            
            
    def set_procedure_docs(self):

        procedures_obj = self.soup.find_all("memberdef", {"kind": "function"})
        if procedures_obj is None: return
                
        for procedure in procedures_obj:
            
            procname = self.get_name(procedure)
            proctype = self.get_type(procedure)
            proc_introsentence= f"{procname} is a {proctype} in {self.modname}."

            argsstring = self.get_argsstring(procedure, procname)            
            proc_description = self.get_parblock_sentences(procedure, procname)
            args = self.get_params_as_sentences(procedure, procname)
            inbodydescription = self.get_inbodydescription_as_sentences(procedure, procname)
                                    
            self.procdocs[procname] = {
                "document": f"{proc_introsentence} {proc_description} {argsstring} {args} {inbodydescription}",
                "metadata": {"source": self.modname, "name": procname, "identity": proctype}
            }

                    
    def get_name(self, tagobj):

        name_obj = tagobj.find("name")

        if name_obj is None:
            raise RuntimeError(f"Cannot find name in \n{tagobj}")

        name = name_obj.text.strip()
        print(name)
        return name

    
    def get_type(self, tagobj):

        type_obj = tagobj.find("type")                

        if type_obj is None:
            raise RuntimeError(f"Cannot find type in \n{tagobj}")

        return type_obj.text.strip()

    
    def get_briefdescription_as_sentence(self, tagobj, name):

        briefdescription_obj = tagobj.briefdescription
        if briefdescription_obj is not None:
            briefdescription = briefdescription_obj.text.strip()
            if briefdescription:
                return f"{name} {briefdescription}."

        return f"There is no description for {name}."

        
    def get_argsstring(self, tagobj, name):

        argsstring_obj = tagobj.argsstring
        
        if argsstring_obj is not None:
            argsstring = argsstring_obj.text.strip()
            if argsstring:
                return f"{name} has the following arguments: {argsstring}."

            return f"There are no arguments for {name}."
        

    def get_parblock_sentences(self, tabobj, name):

        parblock_obj = tabobj.parblock

        if parblock_obj is not None:
            parblock = parblock_obj.text.strip()
            if parblock:
                return parblock

        return f"There is no description for {name}."


    def get_params_as_sentences(self, tabobj, name):

        paramitem_objs = tabobj.find_all("parameteritem")
        if paramitem_objs is None: return f"There are are arguments for {name}"

        paramdoc = ""
        for paramitem_obj in paramitem_objs:
            paramdoc += self.get_paramitem(paramitem_obj, name)

        return paramdoc
                        
        
    def get_paramitem(self, tagobj, name):

        paramnamelist_obj = tagobj.parameternamelist
        if paramnamelist_obj is None: raise RuntimeError(f"Cannot get parameternamelist from \n{tagobj}")

        paramname = paramnamelist_obj.text.split()
        inout = paramnamelist_obj["direction"] if "direction" in paramnamelist_obj.attrs else "inout"

        paramdef_obj = tagobj.parameterdescription
        if paramdef_obj is not None:
            paramdef = paramdef_obj.text.strip()
            if paramdef:
                return f"{paramname} is an intent({inout}) variable. {paramname} {paramdef}."

        return f"There is no definition for argument {paramname}."


    def get_inbodydescription_as_sentences(self, tagobj, name):

        inbodydescription_obj = tagobj.inbodydescription

        if inbodydescription_obj is not None:
            inbodydescription = inbodydescription_obj.text.strip()
            if inbodydescription:
                return  f"The following occurs in {name}: {inbodydescription}"

        return "There are no details about what happens in {name}."

                    
#modxml = namespaceXMLsoup(codebase="FMSCoupler", xmlfile="namespaceatm__land__ice__flux__exchange__mod.xml")
#modxml.set_variable_docs()
#modxml.set_procedure_docs()
#for variable, vardict in modxml.vardocs.items():
#    print(vardict["document"])

