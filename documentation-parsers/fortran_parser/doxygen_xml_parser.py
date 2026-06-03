from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup

from langchain_core.documents import Document


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
            raise IOError(f"xml file '{self.xmlfile}' in directory '{self.xmldir}' does not exist")        

        
    def get_name(self, soup = None, toplevel = False):
        
        if soup is None:
            soup = self.soup

        tag = "name"
        if toplevel:
            tag = "compoundname"

        name = self.get_tag(tag, soup)

        if name is None:
            raise RuntimeError(f"cannot find tag 'compoundname' to set name in {self.soup}.  ")
                
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
        
        parameter_name = parameter_namelist_obj.text.split()[0].strip()

        description = f"{parameter_name} is an intent({inout}) variable.  "
        
        parameter_description = self.get_tag("parameterdescription", parameter_item_obj)
        if parameter_description is not None:
            description += f"{parameter_name} {parameter_description}.  "

        return description


    
class ModuleTopLevelDocument(XMLsoup):

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
        
        overview = f"{self.toplevel_name} is a {self.prog_or_mod} in {self.codebase}.  "
        
        if briefdescription is not None:
            overview += f"{briefdescription}.  "

        if detaileddescription is not None:
            overview += f"{detaileddescription}.  "
        
        return {
            "id": self.toplevel_name,
            "source": self.toplevel_name,
            "type": "overview",
            "description": overview,
        }
                    
    
class ModuleBodyDocument(XMLsoup):

    def __init__(self,
                 codebase: str,
                 prog_or_mod: Literal["module", "program"] = "module",                                                                               
                 xmldir: str|Path = "./docs/xml",
                 xmlfile: str|Path = None):
        super().__init__(codebase, xmldir, xmlfile=xmlfile)
        self.bodyxmlfile = xmlfile

        self.prog_or_mod = prog_or_mod
        self.variables = {}
        self.procedures = {}

        
    def document_module_variables(self):
        """
        Documents module variables.  For example, parses
          module this module
            real(8) :: var1 !< is a variable
            integer :: var2 !< is another variable
          end module this module
        and returns       
          self.variables["var1"] = Document(
                page_content="var1 is a real(8) variable in this module. var1 is a variable.",
                metadata={"source": "this module", "name": "var1"}
                "document": "var1 is a real(8) variable in this module. var1 is a variable.",            
          )
          self.variables["var2"] = Document(
                page_content="var2 is an integer variable in this module. var2 is another variable.",
                metadata={"source": "this module", "name": "var2"}
          )
        """

        variables_obj = self.soup.find_all("memberdef", {"kind": "variable"})

        if variables_obj is None:
            return "There are no module variables in this module.  "
        
        for variable in variables_obj:
            
            varname = self.get_name(variable)
            vartype = self.get_tag("type", variable)
            briefdescription = self.get_tag("briefdescription", variable)

            var_description = f"{varname} is a {self.prog_or_mod} variable in {self.toplevel_name}.  "

            if vartype is not None:
                var_description += f"{varname} is a {vartype}.  "

            if briefdescription is not None:
                var_description += f"{varname} {briefdescription}.  "
            
            self.variables[varname] = Document(
                    page_content=var_description,
                    metadata={"source": self.toplevel_name, "name": varname},
                )

                    
    def document_procedures(self):
        """
        Documents procedures.  For example, parses
          module this_module
            contains
              !> \parblock
              !! Subroutine this_subroutine is an example.
              !> \endparblock
              subroutine this_subroutine(arg1, arg2)
                real(8), intent(in) :: arg1 !< is just an example variable
                integer, intent(out) :: arg2 !< is an another example variable
                !> checks if arg1 > 10
                if(arg1 < 10) call mpp_error(...)
                !> assigns arg2
                arg2 = arg1
              end subroutine this_subroutine
          end module this_module
        and returns
          self.procedures["this_subroutine"] = Document(
                page_content = "Subroutine this_subroutine is an example. 
                             this_subroutine has these subroutine arguments: arg1, arg2. 
                             arg1 is an intent(in) variable. arg1 is an example variable. 
                             arg2 is an intent(out) variable. arg2 is an another example variable.
                             In this_subroutine, the following occurs: 
                             checks if arg1 > 10. assigns arg2 to arg1.",
                metadata={"source": "this_module", "name": "this_subroutine", "identity": "subroutine"},
          )
        """

        procedures_obj = self.soup.find_all("memberdef", {"kind": "function"})
        if procedures_obj is None:
            return "There are no procedures in this module"

        documents = {}
        for procedure in procedures_obj:
            
            procname = self.get_name(procedure) 
            proctype = self.get_tag("type", procedure) #subroutine or function
            argsstring = self.get_tag("argsstring", procedure)
            parameters_description = self.get_parameters_description(procedure)
            briefdescription = self.get_tag("briefdescription", procedure)
            detaileddescription = self.get_tag("parblock", procedure)            
            inbodydescription = self.get_tag("inbodydescription", procedure)
            
            if proctype is None:
                proctype = "procedure"

            procedure_description= f"{procname} is a {proctype} in {self.toplevel_name}.  "
            
            if argsstring is None:
                procedure_description += f"{procname} does not have any {proctype} arguments.  "
            else:
                procedure_description += f"{procname} has these {proctype} arguments:  {argsstring}.  "

            if parameters_description is not None:
                procedure_description += parameters_description

            if briefdescription is not None:
                procedure_description += briefdescription

            if detaileddescription is not None:
                procedure_description += detaileddescription

            if inbodydescription is not None:
                procedure_description += f"In {procname}, the following occurs: {inbodydescription}.  "
                
            self.procedures[procname] = Document(
                page_content=procedure_description,
                metadata={"source": self.toplevel_name, "name": procname},
            )

def test():                   
    modxml = ModuleBodyDocument(codebase="FMSCoupler", xmlfile="namespaceatm__land__ice__flux__exchange__mod.xml")
    modxml.document_module_variables()
    modxml.document_procedures()
    print(modxml.variables)
    print(modxml.procedures)
