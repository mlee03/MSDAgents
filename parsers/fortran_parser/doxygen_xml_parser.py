from pathlib import Path
from pydantic import BaseModel

import bs4


def _clean(text_to_clean: str|bs4.element.Tag) -> str:
    """
    Returns text inside tag and nested tags if text_to_clean is not None.
    Else, returns empty string.
    """
    if isinstance(text_to_clean, str):
        return text_to_clean.strip()
    elif isinstance(text_to_clean, bs4.element.Tag):
        return text_to_clean.get_text().strip() if text_to_clean is not None else ""
    else:
        return ""


class XMLsoup():

    """
    Returns a BeautifulSoup object for the given XML file.
    """

    @staticmethod
    def get(xmldir: str|Path, xmlfile: str|Path):

        if xmlfile is None:
            raise IOError("xmlfile not specified")

        xmlfile = Path(xmldir)/Path(xmlfile)

        if xmlfile.exists():
            with open(xmlfile, "r") as openedfile:
                return bs4.BeautifulSoup(openedfile, "xml")
        else:
            raise FileNotFoundError(f"xml file '{xmlfile}' in directory '{xmldir}' does not exist")


class VariableData(BaseModel):

    """fields for module variables"""

    name: str
    type_: str
    description: str

class ProcedureData(BaseModel):

    """fields for subroutines and functions"""
    name: str
    type_: str
    arguments_list: list|str
    arguments_description: dict[str, VariableData]
    briefdescription: str
    detaileddescription: str
    inbodydescription: list|str


class ModuleVariableParser():

    def __init__(self):
        self.variables_dict = {}
    
    def get(self, soup):

        """
        Documents module variables.  For example, parses

          module this module
            real(8) :: var1 !< is a variable
            integer :: var2 !< is another variable
          end module this module
        
        All variable descriptions are assumed to be less than ~400 tokens.
        """

        variablesoup = soup.doxygen.find("sectiondef", {"kind": "var"})
        if variablesoup is None:            
            print("No module variables")
            return

        variables = variablesoup.find_all("memberdef", {"kind": "variable"})

        if variables:
            for variable in variables:
                varname = _clean(variable.find("name"))
                self.variables_dict[varname] = VariableData(
                    name = varname, 
                    type_ = _clean(variable.type),
                    description = _clean(variable.briefdescription)
                )

        return self.variables_dict

    def get_direct_tag_to_string(self, tag, soup = None):

        """
        Returns the text of a direct child tag without descending into nested
        members that may contain same-named tags earlier in the document.
        """

        if soup is None:
            soup = self.soup

        tagobj = soup.find(tag, recursive=False)

        if tagobj is not None:
            tagstr = tagobj.text.strip()
            if tagstr:
                return tagstr

        return ""


class ProcedureParser():
    
    def __init__(self):
        self.procedures_dict = {}

    def get(self, soup):
    
        """
        Documents subroutines and functions.
        """
        functionsoup = soup.doxygen.find("sectiondef", {"kind": "func"})
        if functionsoup is None:
            return 

        procedures = functionsoup.find_all("memberdef", {"kind": "function"})
        
        if procedures:    
            for procedure in procedures:            
                procname = _clean(procedure.find("name"))  
                procedure_dict = ProcedureData(
                    name = procname,
                    type_ = _clean(procedure.find("type")),
                    arguments_list = _clean(procedure.argsstring).strip("()"),
                    arguments_description = self.get_arguments_description(procedure),
                    briefdescription = _clean(procedure.briefdescription),
                    detaileddescription = _clean(self.get_detaileddescription(procedure)),
                    inbodydescription = self.get_inbodydescription_list(procedure)
                )          
                self.procedures_dict[procname] = procedure_dict

        return self.procedures_dict

    def get_arguments_description(self, soup):

        """
        Returns the parameters for a subroutine
        <parameterlist>
            <parameteritem>
                <parameternamelist>
                    <parametername direction="in">time</parametername>
                </parameternamelist>
                <parameterdescription>
                    <para>is the current model time</para>
                </parameterdescription>
            </parameteritem>
        </parameterlist>
        """

        argument_dict = {}
        arguments = soup.find_all("parameteritem")

        if arguments:
            for argument in arguments:
                namelist = argument.parameternamelist
                name = namelist.text.strip()
                argument_dict[name] = VariableData(
                    name = name,
                    type_ = namelist.get("direction", "inout"),
                    description = _clean(argument.parameterdescription)
                )

        return argument_dict

    def get_detaileddescription(self, soup):

        """
        Parses
        <briefdescription>
            <para><parblock>Brief description.</parblock></para>
        </briefdescription>
        <detaileddescription>
            <para><parblock>Long description.</parblock></para>
        </detaileddescription>
        """
        try:
            return _clean(soup.detaileddescription.parblock)
        except Exception as e:
            return ""
        

    def get_inbodydescription_list(self, soup):
        
        """
        Parses
        <inbodydescription>
        <para><parblock><para>INITIALIZE MODULE-LEVEL VARIABLES. </para></parblock></para>
        <para><parblock><para>GET FILE UNIT FOR STDOUT AND STDLOG FOR INTERNAL LOGGING PURPOSES </para></parblock></para>
        </inbodydescription>
        """

        inbodydescription = []

        inbodydescription_obj = soup.find("inbodydescription")

        if inbodydescription_obj:
            for step in inbodydescription_obj.find_all("para"):
                if step.text.strip():
                    inbodydescription.append(f"{step.text.strip()}.")

        return inbodydescription

    
class TopLevelDocParser():

    def __init__(self):

        self.description_dict = {
            "brief": "",
            "detailed": "",
            "overview": ""
        }
    
    def get(self, soup):
        
        for child in soup.doxygen.compounddef.children:
            if child.name == "briefdescription":
                self.description_dict["brief"] = _clean(child)
            if child.name == "detaileddescription":
                self.description_dict["detailed"] = _clean(child)        

        self.description_dict["overview"] = self.description_dict["brief"] + "  " + self.description_dict["detailed"]
        return self.description_dict  
    

class InterfaceParser(XMLsoup):

    def __init__(self, xmldir: str|Path):

        self.interface_dict = {}
        self.xmldir = xmldir

    def get_interface_files(self, soup):
        
        innerclasses = soup.find_all("innerclass")
        if innerclasses:
            filenames = [_clean(innerclass.get("refid"))+".xml" for innerclass in innerclasses]
            return filenames

    def get(self, soup):

        xmlfiles = self.get_interface_files(soup)
        if xmlfiles is None:
            return self.interface_dict
        
        for xmlfile in xmlfiles:
            xmlsoup = XMLsoup.get(self.xmldir, xmlfile)
            name = _clean(xmlsoup.find("compoundname").text.split("::")[1])
            self.interface_dict[name] = {
                "name": name,
                "members": [_clean(subroutine.text) for subroutine in xmlsoup.find_all("definition")],
                "overview": TopLevelDocParser().get(xmlsoup)["overview"]
            }
        return self.interface_dict        


class ModuleDocument():

    def __init__(self, xmldir: str|Path, group_xmlfile: str|Path):

        self.xmldir = Path(xmldir)
        self.group_xmlfile = Path(group_xmlfile)
        self.xmlsoup = XMLsoup.get(xmldir=xmldir, xmlfile=group_xmlfile)
        self.module_name = None
        self.overview = None
        self.variables = None
        self.procedures = None

        self.interfaces = None

        self.mdfile = []

    def populate(self):

        self.module_name = _clean(self.xmlsoup.doxygen.compoundname)
        self.overview = TopLevelDocParser().get(self.xmlsoup)
        self.variables = ModuleVariableParser().get(self.xmlsoup)
        self.procedures = ProcedureParser().get(self.xmlsoup)
        self.interfaces = InterfaceParser(self.xmldir).get(self.xmlsoup)

    def convert_to_markdown(self):

        self.mdfile.append(f"# Module: {self.module_name}\n")
                
        if self.overview:
            self.mdfile.append(f"{self.overview["overview"]}\n")
            self.mdfile.append("\n")
        
        if self.variables:
            self.mdfile.append("## Module variables\n")
            self.mdfile.append("\n")
            for varname, varinfo in self.variables.items():
                vartype = varinfo.type_ if varinfo.type_ else "unknown"
                vardescript = varinfo.description
                if not vardescript:
                    vardescript = "No description"
                self.mdfile.append(f"### {varname}\n")
                self.mdfile.append(f"- name:  {varname}\n")
                self.mdfile.append(f"- type:  {vartype} variable\n")
                self.mdfile.append(f"- description:  {vardescript}\n")
                self.mdfile.append("\n")
        
        if self.procedures:
            self.mdfile.append(f"## Subroutines and functions\n")
            self.mdfile.append("\n")
            for procname, procinfo in self.procedures.items():
                proctype = procinfo.type_
                
                self.mdfile.append(f"### {proctype}: {procname}\n")
                self.mdfile.append(f"- name:  {procname}\n")
                self.mdfile.append(f"- type:  {proctype}\n")
                self.mdfile.append("\n")
                # description
                description = procinfo.detaileddescription if procinfo.detaileddescription else "No description"
                self.mdfile.append(f"- description:  {description}\n")
                self.mdfile.append("\n")
                # arguments
                if procinfo.arguments_list:
                    self.mdfile.append(f"- arguments:  {procinfo.arguments_list}.")
                if procinfo.arguments_description:
                    arguments = [f"{argname} ({arginfo.type_}) {arginfo.description}" for argname, arginfo in procinfo.arguments_description.items()]
                    self.mdfile.append(".  ".join(arguments))
                    self.mdfile.append("\n\n")
                inbodydescription = procinfo.inbodydescription
                if inbodydescription:
                    self.mdfile.append("- additional description:")
                    for step in inbodydescription:
                        self.mdfile.append(f"{step}")
                self.mdfile.append("\n\n")
        
        if self.interfaces:            
            self.mdfile.append(f"## Interfaces\n")
            for interfacename, interfaceinfo in self.interfaces.items():
                self.mdfile.append(f"### {interfacename}\n")
                self.mdfile.append(f"- name:  {interfacename}\n")
                self.mdfile.append(f"- description:  {interfaceinfo['overview']}\n")
                self.mdfile.append(f"- members:  {', '.join(interfaceinfo['members'])}\n")
                self.mdfile.append("\n")

    def write_markdown(self, output_dir: str|Path, output_file: str|Path = None, create_dir: bool = True):

        self.convert_to_markdown()

        output_file_ = f"{self.module_name}.md" if output_file is None else output_file

        if not Path(output_dir).is_dir():
            if create_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            else:
                raise RuntimeError(f"Directory {output_dir} does not exist")

        with open(Path(output_dir)/output_file_, "w", encoding="utf-8") as f:
            f.write("".join(self.mdfile))

        return output_file_


if __name__ == "__main__":
    xmldir = "/home/Mikyung.Lee/chatbot/fmscoupler-revisions/fms/FMS/docs/xml/"
    xmlfile = "group__horiz__interp__mod.xml"
    module = ModuleDocument(xmldir, xmlfile)
    module.populate()
    module.convert_to_markdown()
    module.write_markdown("./")
