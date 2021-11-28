# Origianl parser and related helper functions code from https://gitlab.com/mattbas/python-lottie
# SVG parse using https://gitlab.com/mattbas/python-lottie/. 
# Change to original code : Generating Lottie using pydantic based object model.

import enum
from xml.etree import ElementTree


class Handler:
    ns_map = {
        "dc": "http://purl.org/dc/elements/1.1/",
        "cc": "http://creativecommons.org/ns#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "svg": "http://www.w3.org/2000/svg",
        "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
        "inkscape": "http://www.inkscape.org/namespaces/inkscape",
        "xlink": "http://www.w3.org/1999/xlink",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    }

    def init_etree(self):
        for n, u in self.ns_map.items():
            ElementTree.register_namespace(n, u)

    def qualified(self, ns, name):
        return "{%s}%s" % (self.ns_map[ns], name)

    def simplified(self, name):
        for k, v in self.ns_map.items():
            name = name.replace("{%s}" % v, k+":")
        return name

    def unqualified(self, name):
        return name.split("}")[-1]

    def __init__(self):
        self.init_etree()


class NameMode(enum.Enum):
    NoName = 0
    Id = 1
    Inkscape = 2
