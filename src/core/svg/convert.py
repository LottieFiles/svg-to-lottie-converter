# Origianl parser and related code from https://gitlab.com/mattbas/python-lottie
# SVG parse using https://gitlab.com/mattbas/python-lottie/. 
# Change to original code : Generating Lottie using pydantic based object model.

import re
import math
import copy
import collections
import numbers
import json
import random
from enum import Enum

from xml.etree import ElementTree
from model import animation as objects
from model import *
import model

from model import animation

from utils.vector import NVector  #as Vector
from utils.transform import TransformMatrix
from core.shape import Point, Size
from .svgdata import color_table, css_atrrs
from .handler import Handler, NameMode
#from model.bezier import Bezier
from core.shape import Ellipse
from model.properties import Value, MultiDimensional, OffsetKeyframe, MDBezier, ShapeProp
import numpy as np
from .gradients import *
"""
try:
    from ...utils import font
    has_font = True
except ImportError:
    has_font = False
"""


def Vector(*args):
    param = []
    for elem in args:
        param.append(elem)
    p = np.array(param)
    v = NVector(p)
    l = len(p)
    list1 = v[0].tolist()

    l = v[0].tolist()

    ##print ("fucking vec type : ", type(l))
    ##print ("return vec is :", l)

    return l

def formatfloat(x):
    #print (type(x),x)
    if isinstance(x,float):
        return round(x, 2)
        #return "%.3g" % float(x)
    else:
        return x

def pformat(dictionary, function):
    if isinstance(dictionary, dict):
        return type(dictionary)((key, pformat(value, function)) for key, value in dictionary.items())
    if isinstance(dictionary, collections.Container):
        if not isinstance(dictionary,str):
            return type(dictionary)(pformat(value, function) for value in dictionary)
    if isinstance(dictionary, numbers.Number):
        return function(dictionary)

    return dictionary

nocolor = {"none"}
unit_convert = {
        None: 1,           # Default unit (same as pixel)
        'px': 1,           # px: pixel. Default SVG unit
        'em': 10,          # 1 em = 10 px FIXME
        'ex': 5,           # 1 ex =  5 px FIXME
        'in': 96,          # 1 in = 96 px
        'cm': 96 / 2.54,   # 1 cm = 1/2.54 in
        'mm': 96 / 25.4,   # 1 mm = 1/25.4 in
        'pt': 96 / 72.0,   # 1 pt = 1/72 in
        'pc': 96 / 6.0,    # 1 pc = 1/6 in
        '%' :  1 / 100.0   # 1 percent
        }

def _sign(x):
    if x < 0:
        return -1
    return 1


class DefsParent:
    def __init__(self):
        self.items = {}

    def insert(self, dummy, shape):
        self.items[shape.name] = shape

    def __getitem__(self, key):
        return self.items[key]

    def __setitem__(self, key, value):
        self.items[key] = value

    def __contains__(self, key):
        return key in self.items

    @property
    def shapes(self):
        return self


class Parser(Handler):
    def __init__(self, name_mode=NameMode.Inkscape):
        self.init_etree()
        self.name_mode = name_mode
        self.current_color = Vector(0, 0, 0, 1)
        self.gradients = {}
        self.max_time = 0
        self.defs = DefsParent()

    def _get_name(self, element, inkscapequal):
        if self.name_mode == NameMode.Inkscape:
            return element.attrib.get(inkscapequal, element.attrib.get("id"))
        return self._get_id(element)

    def _get_id(self, element):
        if self.name_mode != NameMode.NoName:
            return element.attrib.get("id")
        return None

    def _parse_unit(self, value):
        if not isinstance(value, str):
            return value

        mult = 1
        cmin = 2.54
        if value.endswith("px"):
            value = value[:-2]
        elif value.endswith("in"):
            value = value[:-2]
            mult = self.dpi
        elif value.endswith("pc"):
            value = value[:-2]
            mult = self.dpi / 6
        elif value.endswith("pt"):
            value = value[:-2]
            mult = self.dpi / 72
        elif value.endswith("cm"):
            value = value[:-2]
            mult = self.dpi / cmin
        elif value.endswith("mm"):
            value = value[:-2]
            mult = self.dpi / cmin / 10

        elif value.endswith("vw"):
            value = value[:-2]
            mult = self.animation.width * 0.01
        elif value.endswith("vh"):
            value = value[:-2]
            mult = self.animation.height * 0.01
        elif value.endswith("vmin"):
            value = value[:-4]
            mult = min(self.animation.width, self.animation.height) * 0.01
        elif value.endswith("vmax"):
            value = value[:-4]
            mult = max(self.animation.width, self.animation.height) * 0.01
        elif value.endswith("Q"):
            value = value[:-1]
            mult = self.dpi / cmin / 40

        return float(value) * mult

    def parse_color(self, color):
        return parse_color(color, self.current_color)

    def parse_transform(self, element, group, dest_trans):
        bb = group.bounding_box()
        if not bb.isnull():
            itcx = self.qualified("inkscape", "transform-center-x")
            if itcx in element.attrib:
                cx = float(element.attrib[itcx])
                cy = float(
                    element.attrib[self.qualified(
                        "inkscape", "transform-center-y"
                    )]
                )
                bbx, bby = bb.center()
                cx += bbx
                cy = bby - cy
                dest_trans.anchorPoint.value = Vector(cx, cy)
                dest_trans.position.value = Vector(cx, cy)

        if "transform" not in element.attrib:
            return

        matrix = TransformMatrix()
        read_matrix = False

        for t in re.finditer(
            r"([a-zA-Z]+)\s*\(([^\)]*)\)", element.attrib["transform"]
        ):
            name = t[1]
            params = list(map(float, t[2].strip().replace(",", " ").split()))
            if name == "translate":
                dest_trans.position.value += Vector(
                    params[0],
                    (params[1] if len(params) > 1 else 0),
                )
            elif name == "scale":
                xfac = params[0]
                dest_trans.scale.value[0] = (
                    dest_trans.scale.value[0] / 100 * xfac
                ) * 100
                yfac = params[1] if len(params) > 1 else xfac
                dest_trans.scale.value[1] = (
                    dest_trans.scale.value[1] / 100 * yfac
                ) * 100
            elif name == "rotate":
                ang = params[0]
                x = y = 0
                if len(params) > 2:
                    x = params[1]
                    y = params[2]
                    ap = Vector(x, y)
                    dap = ap - dest_trans.position.value
                    dest_trans.position.value += dap
                    dest_trans.anchor_point.value += dap
                dest_trans.rotation.value = ang
            else:
                read_matrix = True
                self._apply_transform_element_to_matrix(matrix, t)

        if read_matrix:
            dtapv = NVector(
                dest_trans.anchorPoint.value[0], dest_trans.anchorPoint.value[1]
            )
            dtpv = NVector(
                dest_trans.position.value[0], dest_trans.position.value[1]
            )
            dtpv -= dtapv

            #print ("dtpv :", dtpv)
            dest_trans.position.value = dtpv.to_list()
            #dest_trans.position.value -= dest_trans.anchor_point.value
            dest_trans.anchorPoint.value = Vector(0, 0)
            trans = matrix.extract_transform()
            dest_trans.skewAxis.value = math.degrees(trans["skew_axis"])
            dest_trans.skew.value = -math.degrees(trans["skew_angle"])

            t = np.array(trans["translation"])
            tt = np.array(dest_trans.position.value)
            tt += t

            #print ("tt :",tt)

            dest_trans.position.value = tt.tolist()

            dest_trans.rotation.value -= math.degrees(trans["angle"])
            desttransscale = np.array(dest_trans.scale.value)
            transscale = np.array(trans["scale"])
            desttransscale *= transscale
            dest_trans.scale.value = desttransscale.tolist()
            #dest_trans.scale.value *= trans["scale"]

    def parse_style(self, element, parent_style):
        #if "class" in (set(element.attrib.keys())):
        #    print ("classssss...")
        #    print (element.attrib["class"])
        style = parent_style.copy()
        for att in css_atrrs & set(element.attrib.keys()):
            if att in element.attrib:
                style[att] = element.attrib[att]
        if "style" in element.attrib:
            style.update(
                **dict(
                    map(
                        lambda x: map(lambda y: y.strip(), x.split(":")),
                        filter(bool, element.attrib["style"].split(";"))
                    )
                )
            )
        return style

    def apply_common_style(self, style, transform):
        opacity = float(style.get("opacity", 1))
        transform.opacity.value = opacity * 100

    def apply_visibility(self, style, object):
        if style.get("display", "inline") == "none" or style.get(
            "visibility", "visible"
        ) == "hidden":
            object.hidden = True

    def add_shapes(self, element, shapes, shape_parent, parent_style):        
        style = self.parse_style(element, parent_style)

        #layer = objects.layers.ShapeLayer()
        #animation.add_layer(layer)

        group = objects.layers.Group()
        grouptransform = group.transform
        self.apply_common_style(style, grouptransform)
        self.apply_visibility(style, group)
        group.name = self._get_id(element)

        shape_parent.shapes.insert(0, group)
        for shape in shapes:
            group.add_shape(shape)

        self._add_style_shapes(style, group)

        self.parse_transform(element, group, grouptransform)

        return group

    def _add_style_shapes(self, style, group):
        stroke_color = style.get("stroke", "none")
        if stroke_color not in nocolor:
            if stroke_color.startswith("url"):
                stroke = self.get_color_url(
                    stroke_color, objects.GradientStroke, group
                )
                opacity = 1
            else:
                stroke = model.shapes.Stroke()
                color = self.parse_color(stroke_color)
                stroke.color.value = color[:4]
                opacity = color[3]
            stroke.opacity.value = opacity * float(
                style.get("stroke-opacity", 1)
            ) * 100
            group.add_shape(stroke)
            stroke.width.value = self._parse_unit(style.get("stroke-width", 1))
            linecap = style.get("stroke-linecap")
            if linecap == "round":
                stroke.lineCap = model.helpers.LineCap.Round
            elif linecap == "butt":
                stroke.lineCap = model.helpers.LineCap.Butt
            elif linecap == "square":
                stroke.lineCap = model.helpers.LineCap.Square
            linejoin = style.get("stroke-linejoin")
            if linejoin == "round":
                stroke.lineJoin = model.helpers.LineJoin.Round
            elif linejoin == "bevel":
                stroke.lineJoin = model.helpers.LineJoin.Bevel
            elif linejoin in {"miter", "arcs", "miter-clip"}:
                stroke.lineJoin = model.helpers.LineJoin.Miter
            stroke.miterLimit = self._parse_unit(
                style.get("stroke-miterlimit", 0)
            )

        fill_color = style.get("fill", "inherit")
        #print (style.get("fill"))
        if fill_color not in nocolor:
            if fill_color.startswith("url"):
                fill = self.get_color_url(fill_color, model.shapes.GFill, group)
                opacity = 1
            else:
                color = self.parse_color(fill_color)
                fcolor = (Vector(*color[:4]))
                vcolor = list(fcolor)
                fill = model.shapes.Fill()
                fill.color = Value(value=vcolor)
                opacity = color[3]
            opacity *= float(style.get("fill-opacity", 1))
            fill.opacity = Value(value=(opacity * 100))
            group.add_shape(fill)

    def _parseshape_use(self, element, shape_parent, parent_style):
        link = element.attrib[self.qualified("xlink", "href")]
        if link.startswith("#"):
            id = link[1:]
            
            #if id in self.defs:
            #    base = self.defs[id]
            #else:
            #    base_element = self.document.find(".//*[@id='%s']" % id)
            #    base = self.parse_shape(base_element, self.defs)

            base_element = self.document.find(".//*[@id='%s']" % id)
            use_style = self.parse_style(element, parent_style)

            used = model.shapes.Group()
            #baseclone = copy.deepcopy(base)
            #used.add_shape(baseclone)
            shape_parent.add_shape(used)
            used.name = "use"

            used.transform.position.value[0] = float(element.attrib.get("x", 0))
            used.transform.position.value[1] = float(element.attrib.get("y", 0))
            self.parse_transform(element, used, used.transform)
            #shape_parent.shapes.insert(0, used)
            self.parse_shape(base_element, used, use_style)

            return used

    def _parseshape_g(self, element, shape_parent, parent_style):
        group = objects.layers.Group()
        shape_parent.shapes.insert(0, group)
        style = self.parse_style(element, parent_style)
        self.apply_common_style(style, group.transform)
        self.apply_visibility(style, group)
        group.name = self._get_name(
            element, self.qualified("inkscape", "label")
        )
        self.parse_children(element, group, style)
        self.parse_transform(element, group, group.transform)
        if group.hidden:  # Lottie web doesn't seem to support .hd
            group.transform.opacity.value = 0
        return group

    def _parseshape_ellipse(self, element, shape_parent, parent_style):
        ellipse = model.shapes.Ellipse()
        ellipse.position.value = Vector(
            self._parse_unit(element.attrib["cx"]),
            self._parse_unit(element.attrib["cy"])
        )
        ellipse.size.value = Vector(
            self._parse_unit(element.attrib["rx"]) * 2,
            self._parse_unit(element.attrib["ry"]) * 2
        )
        self.add_shapes(element, [ellipse], shape_parent, parent_style)
        return ellipse

    def _parseshape_anim_ellipse(self, ellipse, element, animations):
        self._merge_animations(element, animations, "cx", "cy", "position")
        self._merge_animations(
            element, animations, "rx", "ry", "size",
            lambda x, y: Vector(x, y) * 2
        )
        self._apply_animations(ellipse.position, "position", animations)
        self._apply_animations(ellipse.size, "size", animations)

    def _parseshape_circle(self, element, shape_parent, parent_style):
        ellipse = model.shapes.Ellipse()
        ellipse.position.value = Vector(
            self._parse_unit(element.attrib["cx"]),
            self._parse_unit(element.attrib["cy"])
        )
        r = self._parse_unit(element.attrib["r"]) * 2
        ellipse.size.value = Vector(r, r)
        self.add_shapes(element, [ellipse], shape_parent, parent_style)
        return ellipse

    def _parseshape_anim_circle(self, ellipse, element, animations):
        self._merge_animations(element, animations, "cx", "cy", "position")
        self._apply_animations(ellipse.position, "position", animations)
        self._apply_animations(
            ellipse.size, "r", animations, lambda r: Vector(r, r) * 2
        )

    def _parseshape_rect(self, element, shape_parent, parent_style):
        rect = model.shapes.Rect()
        w = self._parse_unit(element.attrib.get("width", 0))
        h = self._parse_unit(element.attrib.get("height", 0))

        try:
            x = self._parse_unit(element.attrib.get("x", 0)) + w / 2
        except:
            x = 0 + w / 2

        try:
            y = self._parse_unit(element.attrib.get("y", 0)) + h / 2
        except:
            y = 0 + h / 2

        rect.position.value = Vector(x, y)
        rect.size.value = Vector(w, h)
        rx = self._parse_unit(element.attrib.get("rx", 0))
        ry = self._parse_unit(element.attrib.get("ry", 0))
        rect.rounded.value = (rx + ry) / 2
        self.add_shapes(element, [rect], shape_parent, parent_style)
        return rect

    def _parseshape_anim_rect(self, rect, element, animations):
        self._merge_animations(
            element, animations, "width", "height", "size",
            lambda x, y: Vector(x, y)
        )
        self._apply_animations(rect.size, "size", animations)
        self._merge_animations(element, animations, "x", "y", "position")
        self._merge_animations(
            element, animations, "position", "size", "position",
            lambda p, s: p + s / 2
        )
        self._apply_animations(rect.position, "position", animations)
        self._merge_animations(
            element, animations, "rx", "ry", "rounded", lambda x, y: (x + y) / 2
        )
        self._apply_animations(rect.rounded, "rounded", animations)

    def _parseshape_line(self, element, shape_parent, parent_style):
        line = model.shapes.Path()
        line.shape.value.add_point(
            Vector(
                self._parse_unit(element.attrib["x1"]),
                self._parse_unit(element.attrib["y1"])
            )
        )
        line.shape.value.add_point(
            Vector(
                self._parse_unit(element.attrib["x2"]),
                self._parse_unit(element.attrib["y2"])
            )
        )
        return self.add_shapes(element, [line], shape_parent, parent_style)

    def _parseshape_anim_line(self, group, element, animations):
        line = group.shapes[0]
        self._merge_animations(element, animations, "x1", "y1", "p1")
        self._merge_animations(element, animations, "x2", "y2", "p2")
        #todo : fix 
       
        #self._apply_animations(line.shape.value.vertices[0], "p1", animations)
        #self._apply_animations(line.shape.value.vertices[1], "p2", animations)


    def _handle_poly(self, element):
        line = model.shapes.Path()
        coords = list(
            map(float, element.attrib["points"].replace(",", " ").split())
        )
        for i in range(0, len(coords), 2):
            line.shape.value.add_point(coords[i:i + 2])
        return line

    def _parseshape_polyline(self, element, shape_parent, parent_style):
        line = self._handle_poly(element)
        return self.add_shapes(element, [line], shape_parent, parent_style)

    def _parseshape_polygon(self, element, shape_parent, parent_style):
        line = self._handle_poly(element)
        line.shape.value.close()
        return self.add_shapes(element, [line], shape_parent, parent_style)

    def _parseshape_path(self, element, shape_parent, parent_style):
        d_parser = PathDParser(element.attrib.get("d", ""))
        d_parser.parse()
        paths = []
        for path in d_parser.paths:
            p = model.shapes.Path()

            for x in range(len(path.inPoint)):
                if isinstance(path.inPoint[x], NVector):
                    path.inPoint[x] = [path.inPoint[x][0], path.inPoint[x][1]]

            for x in range(len(path.outPoint)):
                if isinstance(path.outPoint[x], NVector):
                    path.outPoint[x] = [
                        path.outPoint[x][0], path.outPoint[x][1]
                    ]

            for x in range(len(path.vertices)):
                if isinstance(path.vertices[x], NVector):
                    path.vertices[x] = [
                        path.vertices[x][0], path.vertices[x][1]
                    ]

            p.shape.value = path
            paths.append(p)
        #if len(d_parser.paths) > 1:
        #paths.append(objects.shapes.Merge())
        return self.add_shapes(element, paths, shape_parent, parent_style)

    def parse_children(self, element, shape_parent, parent_style):
        #print (parent_style)

        for child in element:
            tag = self.unqualified(child.tag)
            if not self.parse_shape(child, shape_parent, parent_style):
                handler = getattr(self, "_parse_" + tag, None)
                if handler:
                    handler(child)

    def parse_shape(self, element, shape_parent, parent_style):
        handler = getattr(
            self, "_parseshape_" + self.unqualified(element.tag), None
        )
        if handler:
            out = handler(element, shape_parent, parent_style)
            self.parse_animations(out, element)
            if element.attrib.get("id"):
                self.defs.items[element.attrib["id"]] = out
            return out
        return None

    def parse_etree(self, etree, layer_frames=0, *args, **kwargs):
        #print ("------- parse_etree --------")
        animation = objects.Animation(*args, **kwargs)
        self.animation = animation
        self.max_time = 0
        self.document = etree

        svg = etree.getroot()

        self.dpi = float(
            svg.attrib.get(self.qualified("inkscape", "export-xdpi"), 96)
        )

        if "width" in svg.attrib and "height" in svg.attrib:
            animation.width = int(round(self._parse_unit(svg.attrib["width"])))
            animation.height = int(
                round(self._parse_unit(svg.attrib["height"]))
            )
        else:
            _, _, animation.width, animation.height = map(
                float, svg.attrib["viewBox"].split(" ")
            )
        animation.name = self._get_name(
            svg, self.qualified("sodipodi", "docname")
        )

        if layer_frames:
            for frame in svg:
                if self.unqualified(frame.tag) == "g":
                    layer = objects.layers.ShapeLayer()
                    layer.startFrame = self.max_time
                    animation.add_layer(layer)
                    self._parseshape_g(frame, layer, {})
                    self.max_time += layer_frames
                    layer.endFrame = self.max_time
            animation.out_point = self.max_time
        else:
            layer = objects.layers.ShapeLayer()
            animation.add_layer(layer)
            self.parse_children(svg, layer, {})
            if self.max_time:
                animation.endFrame = self.max_time
                for layer in animation.layers:
                    layer.endFrame = self.max_time

        if "viewBox" in svg.attrib:
            vbx, vby, vbw, vbh = map(float, svg.attrib["viewBox"].split())
            if vbx != 0 or vby != 0 or vbw != animation.width or vbh != animation.height:
                for layer in animation.layers:
                    neg = Vector(vbx, vby)
                    neglist = [-x for x in neg]
                    layer.transform.position.value = neglist  #-Vector(vbx, vby)
                    sx = ((animation.width / vbw) * 100)
                    sy = ((animation.height / vbh) * 100)
                    layer.transform.scale.value = [sx, sy]

        return animation

    def _parse_defs(self, element):
        self.parse_children(element, self.defs, {})

    def _apply_transform_element_to_matrix(self, matrix, t):
        name = t[1]
        params = list(map(float, t[2].strip().replace(",", " ").split()))
        if name == "translate":
            matrix.translate(
                params[0],
                (params[1] if len(params) > 1 else 0),
            )
        elif name == "scale":
            xfac = params[0]
            yfac = params[1] if len(params) > 1 else xfac
            matrix.scale(xfac, yfac)
        elif name == "rotate":
            ang = params[0]
            x = y = 0
            if len(params) > 2:
                x = params[1]
                y = params[2]
                matrix.translate(-x, -y)
                matrix.rotate(math.radians(ang))
                matrix.translate(x, y)
            else:
                matrix.rotate(math.radians(ang)) #angle
        elif name == "skewX":
            matrix.skew(math.radians(params[0]), 0)
        elif name == "skewY":
            matrix.skew(0, math.radians(params[0]))
        elif name == "matrix":
            m = TransformMatrix()
            m.a, m.b, m.c, m.d, m.tx, m.ty = params

            matrix *= m

    def _transform_to_matrix(self, transform):
        matrix = TransformMatrix()

        for t in re.finditer(r"([a-zA-Z]+)\s*\(([^\)]*)\)", transform):
            self._apply_transform_element_to_matrix(matrix, t)

        return matrix

    def _gradient(self, element, grad):
        # TODO parse gradientTransform
        grad.matrix = self._transform_to_matrix(
            element.attrib.get("gradientTransform", "")
        )

        id = element.attrib["id"]
        if id in self.gradients:
            grad.colors = self.gradients[id].colors
        grad.parse_attrs(element.attrib)
        href = element.attrib.get(self.qualified("xlink", "href"))
        if href:
            srcid = href.strip("#")
            if srcid in self.gradients:
                src = self.gradients[srcid]
            else:
                src = grad.__class__()
                self.gradients[srcid] = src
            grad.colors = src.colors

        for stop in element.findall("./%s" % self.qualified("svg", "stop")):
            off = float(stop.attrib["offset"].strip("%"))
            if stop.attrib["offset"].endswith("%"):
                off /= 100
            style = self.parse_style(stop, {})
            color = self.parse_color(style["stop-color"])
            if "stop-opacity" in style:
                color[3] = float(style["stop-opacity"])
            grad.add_color(off, color)
        self.gradients[id] = grad

    def _parse_linearGradient(self, element):
        self._gradient(element, SvgLinearGradient())

    def _parse_radialGradient(self, element):
        self._gradient(element, SvgRadialGradient())

    def get_color_url(self, color, gradientclass, shape):
        match = re.match(r"""url\(['"]?#([^)'"]+)['"]?\)""", color)
        if not match:
            return None
        id = match[1]
        if id not in self.gradients:
            return None
        grad = self.gradients[id]
        outgrad = gradientclass()
        grad.to_lottie(outgrad, shape)
        if self.name_mode != NameMode.NoName:
            grad.name = id
        return outgrad

    ## \todo Parse single font property, fallback family etc
    def _parse_text_style(self, style, font_style=None):
        if "font-family" in style:
            font_style.query.family(style["font-family"])

        if "font-style" in style:
            if style["font-style"] == "oblique":
                font_style.query.custom("slant", 110)
            elif style["font-style"] == "italic":
                font_style.query.custom("slant", 100)

        if "font-weight" in style:
            if style["font-weight"] in {"bold", "bolder"}:
                font_style.query.weight(200)
            elif style["font-weight"] == "lighter":
                font_style.query.weight(50)
            elif style["font-weight"].isdigit():
                font_style.query.css_weight(int(style["font-weight"]))

        if "font-size" in style:
            fz = style["font-size"]
            fz_names = {
                "xx-small": 8,
                "x-small": 16,
                "small": 32,
                "medium": 64,
                "large": 128,
                "x-large": 256,
                "xx-large": 512,
            }
            if fz in fz_names:
                font_style.size = fz_names[fz]
            elif fz == "smaller":
                font_style.size /= 2
            elif fz == "larger":
                font_style.size *= 2
            elif fz.endswith("px"):
                font_style.size = float(fz[:-2])
            elif fz.isnumeric():
                font_style.size = float(fz)

    def _parse_text_elem(self, element, style, group, font_style):
        self._parse_text_style(style, font_style)

        if "x" in element.attrib or "y" in element.attrib:
            font_style.position = Vector(
                float(element.attrib["x"]),
                float(element.attrib["y"]),
            )

        if element.text:
            group.add_shape(font.FontShape(element.text, font_style)).refresh()
        for child in element:
            if child.tag == self.qualified("svg", "tspan"):
                self._parseshape_text(child, group, font_style.clone())
            if child.tail:
                group.add_shape(font.FontShape(child.text,
                                               font_style)).refresh()

    def _parseshape_text(self, element, shape_parent, font_style=None):
        group = objects.Group()
        style = self.parse_style(element)
        self.apply_common_style(style, group.transform)
        self.apply_visibility(style, group)
        group.name = self._get_id(element)
        if has_font:
            if font_style is None:
                font_style = font.FontStyle("", 64)
            self._parse_text_elem(element, style, group, font_style)

        style.setdefault("fill", "none")
        self._add_style_shapes(style, group)

        if element.tag == self.qualified("svg", "text"):
            dx = 0
            dy = 0

            ta = style.get("text-anchor", style.get("text-align", ""))
            if ta == "middle":
                dx -= group.bounding_box().width / 2
            elif ta == "end":
                dx -= group.bounding_box().width

            if dx or dy:
                ng = objects.Group()
                ng.add_shape(group)
                group.transform.position.value.x += dx
                group.transform.position.value.y += dy
                group = ng

        shape_parent.shapes.insert(0, group)
        self.parse_transform(element, group, group.transform)

    def parse_animations(self, lottie, element):
        animations = {}
        for child in element:
            if self.unqualified(child.tag) == "animate":
                att = child.attrib["attributeName"]

                from_val = child.attrib["from"]
                if att == "d":
                    ## @todo
                    continue
                else:
                    from_val = float(from_val)
                    if "to" in child.attrib:
                        to_val = float(child.attrib["to"])
                    elif "by" in child.attrib:
                        to_val = float(child.attrib["by"]) + from_val

                begin = self.parse_animation_time(child.attrib.get("begin", 0)
                                                 ) or 0
                if "dur" in child.attrib:
                    end = (self.parse_animation_time(child.attrib["dur"]) or
                           0) + begin
                elif "end" in child.attrib:
                    end = self.parse_animation_time(child.attrib["dur"]) or 0
                else:
                    continue

                if att not in animations:
                    animations[att] = {}
                animations[att][begin] = from_val
                animations[att][end] = to_val
                if self.max_time < end:
                    self.max_time = end

        tag = self.unqualified(element.tag)
        handler = getattr(self, "_parseshape_anim_" + tag, None)
        if handler:
            handler(lottie, element, animations)

    def parse_animation_time(self, value):
        if not value:
            return None
        try:
            seconds = 0
            if ":" in value:
                mult = 1
                for elem in reversed(value.split(":")):
                    seconds += float(elem) * mult
                    mult *= 60
            elif value.endswith("s"):
                seconds = float(value[:-1])
            elif value.endswith("ms"):
                seconds = float(value[:-2]) / 1000
            elif value.endswith("min"):
                seconds = float(value[:-3]) * 60
            elif value.endswith("h"):
                seconds = float(value[:-1]) * 60 * 60
            else:
                seconds = float(value)
            return seconds * self.animation.frame_rate
        except ValueError:
            pass
        return None

    def _merge_animations(
        self, element, animations, val1, val2, dest, merge=Vector
    ):
        if val1 not in animations and val2 not in animations:
            return

        dict1 = list(sorted(animations.pop(val1, {}).items()))
        dict2 = list(sorted(animations.pop(val2, {}).items()))

        x = float(element.attrib[val1])
        y = float(element.attrib[val2])
        values = {}
        while dict1 or dict2:
            if not dict1 or (dict2 and dict1[0][0] > dict2[0][0]):
                t, y = dict2.pop(0)
            elif not dict2 or dict1[0][0] < dict2[0][0]:
                t, x = dict1.pop(0)
            else:
                t, x = dict1.pop(0)
                t, y = dict2.pop(0)

            values[t] = merge(x, y)

        animations[dest] = values

    def _apply_animations(
        self, animatable, name, animations, transform=lambda v: v
    ):
        if name in animations:
            for t, v in animations[name].items():
                animatable.add_keyframe(t, transform(v))


class PathDParser:
    _re = re.compile(
        "|".join(
            (
                r"[a-zA-Z]",
                r"[-+]?[0-9]*\.?[0-9]*[eE][-+]?[0-9]+",
                r"[-+]?[0-9]*\.?[0-9]+",
            )
        )
    )

    def __init__(self, d_string):
        self.path = model.properties.pathBezier()
        self.paths = []
        self.p = Vector(0, 0)
        #print ("p in class PathDParser => ",self.p)
        self.la = None
        self.la_type = None
        self.tokens = list(map(self.d_subsplit, self._re.findall(d_string)))
        self.add_p = True
        self.implicit = "M"

    def d_subsplit(self, tok):
        if tok.isalpha():
            return tok
        return float(tok)

    def next_token(self):
        if self.tokens:
            self.la = self.tokens.pop(0)
            if isinstance(self.la, str):
                self.la_type = 0
            else:
                self.la_type = 1
        else:
            self.la = None

        return self.la

    def next_vec(self):
        x = self.next_token()
        y = self.next_token()
        return [x, y]

    def cur_vec(self):
        x = self.la
        y = self.next_token()
        return Vector(x, y)

    def parse(self):

        self.next_token()

        while self.la is not None:
            if self.la_type == 0:
                parser = "_parse_" + self.la
                self.next_token()
                getattr(self, parser)()
            else:
                parser = "_parse_" + self.implicit
                getattr(self, parser)()

    def _push_path(self):
        self.path = model.properties.pathBezier()
        self.add_p = True

    def _parse_M(self):
        if self.la_type != 1:
            self.next_token()
            return
        self.p = self.cur_vec()
        self.implicit = "L"
        if not self.add_p:
            self._push_path()
        self.next_token()

    def _parse_m(self):
        if self.la_type != 1:
            self.next_token()
            return
        cv = np.array(self.cur_vec()).astype(float)
        p = (np.array(self.p)).astype(float)

        p += cv
        self.p = p.tolist()
        self.implicit = "l"
        if not self.add_p:
            self._push_path()
        self.next_token()

    def _rpoint(self, point, rel=None):
        try:
            p = self.p.value
            self.p = None
            self.p = p
        except:
            self.p = self.p

        if rel is not None:
            sub = rel
        else:
            sub = self.p

        if point is not None:
            difference = []
            zip_object = zip(point, sub)

            for list1_i, list2_i in zip_object:
                difference.append(list1_i - list2_i)

            diff = Value(value=difference)

            return diff.value

        else:
            return Point(0, 0)

    def _do_add_p(self, outp=None):
        if self.add_p:
            self.paths.append(self.path)
            p = copy.deepcopy(self.p)
            self.path.add_point(p, Vector(0, 0), self._rpoint(outp))
            self.add_p = False
        elif outp:
            rp = self.path.vertices[-1]
            self.path.outPoint[-1] = self._rpoint(outp, rp)

    def _parse_L(self):
        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()
        self.p = self.cur_vec()
        pclone = copy.deepcopy(self.p)
        self.path.add_point(pclone, Vector(0, 0), Vector(0, 0))
        self.implicit = "L"
        self.next_token()

    def _parse_l(self):
        if isinstance(self.p, NVector):
            self.p = [self.p[0], self.p[1]]

        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()
        curvec = np.array(self.cur_vec())
        p = self.p
        p = p + curvec
        self.p = p.tolist()
        pclone = copy.deepcopy(self.p)
        self.path.add_point(pclone, Vector(0, 0), Vector(0, 0))
        self.implicit = "l"
        self.next_token()

    def _parse_H(self):
        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()
        self.p[0] = self.la
        pclone = copy.deepcopy(self.p)
        self.path.add_point(pclone, Vector(0, 0), Vector(0, 0))
        self.implicit = "H"
        self.next_token()

    def _parse_h(self):
        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()
        self.p[0] += self.la
        pclone = copy.deepcopy(self.p)
        self.path.add_point(pclone, Vector(0, 0), Vector(0, 0))
        self.implicit = "h"
        self.next_token()

    def _parse_V(self):
        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()
        self.p[1] = self.la
        pclone = copy.deepcopy(self.p)

        self.path.add_point(pclone, Vector(0, 0), Vector(0, 0))
        self.implicit = "V"
        self.next_token()

    def _parse_v(self):
        if isinstance(self.p, NVector):
            self.p = [self.p[0], self.p[1]]
        if isinstance(self.p, np.ndarray):
            p = self.p.tolist()
        else:
            p = self.p

        self.p = p
        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()
        self.p[1] += self.la
        p = copy.deepcopy(self.p)
        self.path.add_point(p, Vector(0, 0), Vector(0, 0))
        self.implicit = "v"
        self.next_token()

    def _parse_C(self):
        if self.la_type != 1:
            self.next_token()
            return
        pout = self.cur_vec()
        self._do_add_p(pout)
        pin = self.next_vec()
        self.p = self.next_vec()
        pclone = copy.deepcopy(self.p)
        pinminusp = (np.array(pin)) - (np.array(self.p))
        self.path.add_point(pclone, pinminusp.tolist(), Vector(0, 0))
        self.implicit = "C"
        self.next_token()

    def _parse_c(self):
        if self.la_type != 1:
            self.next_token()
            return
        curvec = np.array(self.cur_vec())
        p = np.array(self.p)
        pout = (p + curvec).tolist()
        self._do_add_p(pout)
        pin = np.array(self.p) + np.array(self.next_vec())
        nvc = np.array(self.next_vec())
        p = np.array(self.p)
        p = p + nvc
        self.p = p
        inp = ((np.array(pin) - np.array(p))).tolist()

        p = copy.deepcopy(self.p)
        self.path.add_point(p, inp, Vector(0, 0))
        self.implicit = "c"
        self.next_token()

    def _parse_S(self):
        if self.la_type != 1:
            self.next_token()
            return
        pin = self.cur_vec()
        self._do_add_p()
        handle = self.path.inPoint[-1]
        h = [-x for x in handle]
        self.path.outPoint[-1] = h
        self.p = self.next_vec()
        p = copy.deepcopy(self.p)
        self.path.add_point(
            p, (np.array(pin) - np.array(self.p)).tolist(), Vector(0, 0)
        )
        self.implicit = "S"
        self.next_token()

    def _parse_s(self):
        if isinstance(self.p, NVector):
            self.p = [self.p[0], self.p[1]]

        if self.la_type != 1:
            self.next_token()
            return

        curvec = np.array(self.cur_vec())
        p = np.array(self.p)
        pin = (curvec + p).tolist()
        self._do_add_p()
        handle = self.path.inPoint[-1]
        h = [-x for x in handle]
        self.path.outPoint[-1] = h
        nextvec = np.array(self.next_vec())
        p = np.array(self.p)
        p += nextvec
        self.p = p.tolist()
        pinminusp = (np.array(pin) - np.array(self.p)).tolist()
        pclone = copy.deepcopy(self.p)
        self.path.add_point(pclone, pinminusp, Vector(0, 0))
        self.implicit = "s"
        self.next_token()

    def _parse_Q(self):
        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()
        pin = self.cur_vec()
        self.p = self.next_vec()

        pinminusp = (np.array(pin) - np.array(self.p)).tolist()
        pclone = copy.deepcopy(self.p)

        self.path.add_point(pclone, pinminusp, Vector(0, 0))
        self.implicit = "Q"
        self.next_token()

    def _parse_q(self):
        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()

        curvec = np.array(self.cur_vec())
        p = np.array(self.p)
        pin = (p + curvec).tolist()
        nextvec = np.array(self.next_vec())
        p = np.array(self.p)
        p += nextvec
        self.p = p.tolist()
        pinminusp = (np.array(pin) - np.array(self.p)).tolist()
        pclone = copy.deepcopy(self.p)

        self.path.add_point(pclone, pinminusp, Vector(0, 0))
        self.implicit = "q"
        self.next_token()

    def _parse_T(self):
        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()
        handle = self.p - self.path.inPoint[-1]
        self.p = self.cur_vec()
        self.path.add_point(self.p.clone(), (handle - self.p), Vector(0, 0))
        self.implicit = "T"
        self.next_token()

    def _parse_t(self):
        if self.la_type != 1:
            self.next_token()
            return
        self._do_add_p()
        h = [-x for x in (self.path.inPoint[-1])]
        handle = np.array(h) + np.array(self.p)
        handle = handle - np.array(self.p)
        curvec = np.array(self.cur_vec())
        selfp = np.array(self.p)
        selfp += curvec
        self.p = selfp.tolist()
        pclone = copy.deepcopy(self.p)
        self.path.add_point(pclone, (handle.tolist()), Vector(0, 0))
        self.implicit = "t"
        self.next_token()

    def _parse_A(self):
        if self.la_type != 1:
            self.next_token()
            return
        r = self.cur_vec()
        xrot = self.next_token()
        large = self.next_token()
        sweep = self.next_token()
        dest = self.next_vec()
        self._do_arc(r[0], r[1], xrot, large, sweep, dest)
        self.implicit = "A"
        self.next_token()

    def _do_arc(self, rx, ry, xrot, large, sweep, dest):
        if isinstance(dest, np.ndarray):
            dest = dest.tolist()
        self._do_add_p()
        if isinstance(self.p, np.ndarray):
            self.p = [self.p[0], self.p[1]]
        if isinstance(self.p, NVector):
            self.p = [self.p[0], self.p[1]]
        if (self.p) == (dest):
            return

        if rx == 0 or ry == 0:
            self.p = dest
            pclone = copy.deepcopy(self.p)
            self.path.add_point(pclone, Vector(0, 0), Vector(0, 0))
            return

        dest = NVector(dest[0], dest[1])
        ellipse, theta1, deltatheta = Ellipse.from_svg_arc(
            NVector(self.p[0], self.p[1]), rx, ry, xrot, large, sweep, dest
        )
        points = ellipse.to_bezier(theta1, deltatheta)
        self._do_add_p()
        self.path.outPoint[-1] = points[0].out_tangent
        for point in points[1:-1]:
            self.path.add_point(
                point.vertex,
                point.in_tangent,
                point.out_tangent,
            )
        self.path.add_point(
            dest.clone(),
            points[-1].in_tangent,
            Vector(0, 0),
        )
        self.p = dest

    def _parse_a(self):
        if isinstance(self.p, NVector):
            self.p = [self.p[0], self.p[1]]

        if self.la_type != 1:
            self.next_token()
            return
        r = self.cur_vec()
        xrot = self.next_token()
        large = self.next_token()
        sweep = self.next_token()
        nvec = self.next_vec()
        dest = np.array(self.p) + np.array(nvec)
        self._do_arc(r[0], r[1], xrot, large, sweep, dest)
        self.implicit = "a"
        self.next_token()

    def _parse_Z(self):
        if self.path.vertices:
            self.p = copy.deepcopy(self.path.vertices[0])
        self.path.close()
        self._push_path()

    def _parse_z(self):
        self._parse_Z()


def parse_svg_etree(etree, layer_frames=0, *args, **kwargs):
    parser = Parser()
    return parser.parse_etree(etree, layer_frames, *args, **kwargs)


def convert_svg_to_lottie_def(file, layer_frames=0, *args, **kwargs):
    try:
        anim = parse_svg_etree(
            ElementTree.parse(file), layer_frames, *args, **kwargs
        )
        
        an = anim
        lottie = animation.Animation()
        lottie.width = int(an.width)
        lottie.height = int(an.height)
        lottie.endFrame = 1

        shapeslen = (len(an.layers[0].shapes))
        #trans = {"ty":"tr"} # an.layers[0].transform
        trans = an.layers[0].transform

        for x in range(shapeslen):
            if (len(an.layers[0].shapes[x].shapes)) > 1 :
                layer = model.layers.ShapeLayer()
                lottie.add_layer(layer)
                layer.endFrame = 60    
                layer.transform = trans
                g = layer.add_shape(model.layers.Group())
                anshape = an.layers[0].shapes[x]
                layer.shapes[0] = anshape
                if anshape.name is None:
                    layer.name = "layer"+str(x)
                else:
                    layer.name = anshape.name

        data = (lottie.json(by_alias=True,exclude_none=True))
        jsondata = json.loads(data)
        optdata = pformat(jsondata, formatfloat)
        return (optdata)
    except:
        return {"error!"}


def convert_svg_to_lottie(file, layer_frames=0, *args, **kwargs):
    try:
        anim = parse_svg_etree(
            ElementTree.parse(file), layer_frames, *args, **kwargs
        )

        lottie = animation.Animation()

        lottie.frameRate = anim.frameRate
        lottie.startFrame = anim.startFrame
        lottie.endFrame = anim.endFrame
        lottie.width = anim.width
        lottie.height = anim.height
        lottie.endFrame = 1

        optishape = []

        def getshapes(group):
            for item in group.shapes:
                if item.type == "gr":
                    #shapes = getshapes(item)
                    getshapes(item)
                else:
                    optishape.append(item)

        l = 0
        for layers in anim.layers:
            for shape in layers.shapes:
                getshapes(shape)
                        
            layer = model.layers.ShapeLayer()
            lottie.add_layer(layer)
            #lottie.layers[l].transform = {"ty":"tr"} #layers.transform
            lottie.layers[l].transform = layers.transform

            newshapes = []
            for eachshape in optishape:
                if not eachshape.type == "tr":
                    newshapes.append(eachshape)
            
            lastshape = (len(newshapes))
            if newshapes[lastshape-1].type == "tr":
                newshapes.pop()
                    
            lottie.layers[l].shapes = newshapes
            l += 1

        data = (lottie.json(by_alias=True,exclude_none=True))
        #data = (lottie.json(by_alias=True,exclude_none=True))
        jsondata = json.loads(data)
        optdata = pformat(jsondata, formatfloat)

        return (optdata)

    except:
        return {"error!"}

def base_convert_svg_to_lottie(file, layer_frames=0, *args, **kwargs):
    #try:
    anim = parse_svg_etree(
        ElementTree.parse(file), layer_frames, *args, **kwargs
    )

    return anim