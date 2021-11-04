# Refer to https://gitlab.com/mattbas/python-lottie

import re
import math
import numpy as np
import matplotlib.colors as colors

from utils.vector import NVector #as Vector
from utils.transform import TransformMatrix
from .svgdata import color_table, css_atrrs



from model import * 
import model

#__all__ = ['color_mode', 'Color']

color_parse_mode = 'RGB'
color_range = (255, 255, 255, 255)

def cVector(*args):
    param =[]
    for elem in args :
        param.append (elem)
    p = np.array(param)
    v = NVector(p)
    l = len(p)
    list1 = v[0].tolist()

    l = v[0].tolist()

    ##print ("fucking vec type : ", type(l))
    ##print ("return vec is :", l)

    return l

nocolor = {"none"}

def hsl_to_rgb(h, s, l):
    if l < 0.5:
        m2 = l * (s + 1)
    else:
        m2 = l + s - l * s
    m1 = l*2 - m2
    r = hue_to_rgb(m1, m2, h+1/3)
    g = hue_to_rgb(m1, m2, h)
    b = hue_to_rgb(m1, m2, h-1/3)
    a = 1
    return [r, g, b, a]


def hue_to_rgb(m1, m2, h):
    if h < 0:
        h += 1
    elif h > 1:
        h -= 1
    if h*6 < 1:
        return m1+(m2-m1)*h*6
    elif h*2 < 1:
        return m2
    elif h*3 < 2:
        return m1+(m2-m1)*(2/3-h)*6

    #print (m1)

    return m1


class SvgGradientCoord:
    def __init__(self, name, comp, value, percent):
        self.name = name
        self.comp = comp
        self.value = value
        self.percent = percent

    def to_value(self, bbox, default=None):
        if self.value is None:
            return default

        if not self.percent:
            return self.value

        if self.comp == "w":
            return (bbox.x2 - bbox.x1) * self.value

        if self.comp == "x":
            return bbox.x1 + (bbox.x2 - bbox.x1) * self.value

        return bbox.y1 + (bbox.y2 - bbox.y1) * self.value

    def parse(self, attr, default_percent):
        if attr is None:
            return
        if attr.endswith("%"):
            self.percent = True
            self.value = float(attr[:-1])/100
        else:
            self.percent = default_percent
            self.value = float(attr)


class SvgGradient:
    def __init__(self):
        self.colors = []
        self.coords = []
        self.matrix = TransformMatrix()


    def add_color(self, offset, color):
        #self.colors.append((offset, color[:3]))
        self.colors.append((offset, color[:4]))

    def to_lottie(self, gradient_shape, shape, time=0):

        for off, col in self.colors:
            gradient_shape.colors.add_color(off, col)

    def add_coord(self, value):
        setattr(self, value.name, value)
        self.coords.append(value)

    def parse_attrs(self, attrib):
        relunits = attrib.get("gradientUnits", "") != "userSpaceOnUse"
        for c in self.coords:
            c.parse(attrib.get(c.name, None), relunits)


class SvgLinearGradient(SvgGradient):
    def __init__(self):
        super().__init__()
        self.add_coord(SvgGradientCoord("x1", "x", 0, True))
        self.add_coord(SvgGradientCoord("y1", "y", 0, True))
        self.add_coord(SvgGradientCoord("x2", "x", 1, True))
        self.add_coord(SvgGradientCoord("y2", "y", 0, True))

    def to_lottie(self, gradient_shape, shape, time=0):
        bbox = shape.bounding_box(time)
        gradient_shape.startPoint.value = self.matrix.apply(NVector(
            self.x1.to_value(bbox),
            self.y1.to_value(bbox),
        ))
        gradient_shape.endPoint.value = self.matrix.apply(NVector(
            self.x2.to_value(bbox),
            self.y2.to_value(bbox),
        ))
        gradient_shape.gradientType = model.shapes.GradientType.Linear

        super().to_lottie(gradient_shape, shape, time)



class SvgRadialGradient(SvgGradient):
    def __init__(self):
        super().__init__()
        self.add_coord(SvgGradientCoord("cx", "x", 0.5, True))
        self.add_coord(SvgGradientCoord("cy", "y", 0.5, True))
        self.add_coord(SvgGradientCoord("fx", "x", None, True))
        self.add_coord(SvgGradientCoord("fy", "y", None, True))
        self.add_coord(SvgGradientCoord("r", "w", 0.5, True))


    def to_lottie(self, gradient_shape, shape, time=0):
        bbox = shape.bounding_box(time)
        cx = self.cx.to_value(bbox)
        cy = self.cy.to_value(bbox)
        gradient_shape.startPoint.value = self.matrix.apply(NVector(cx, cy))
        r = self.r.to_value(bbox)
        gradient_shape.endPoint.value = self.matrix.apply(NVector(cx+r, cy))

        fx = self.fx.to_value(bbox, cx) - cx
        fy = self.fy.to_value(bbox, cy) - cy
        gradient_shape.highlightAngle.value = math.atan2(fy, fx) * 180 / math.pi
        gradient_shape.highlightLength.value = math.hypot(fx, fy)

        gradient_shape.gradientType = shapes.GradientType.Radial

        super().to_lottie(gradient_shape, shape, time)

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def parse_color(color, current_color=cVector(0, 0, 0, 1)):

    if re.match(r"^#[0-9a-fA-F]{6}$", color):
        col = colors.hex2color(color)
        return cVector(col[0],col[1],col[2],1)
        #return cVector(int(color[1:3], 16) / 0xff, int(color[3:5], 16) / 0xff, int(color[5:7], 16) / 0xff, 1)

    if re.match(r"^#[0-9a-fA-F]{3}$", color):
        col = colors.hex2color(color)
        return cVector(col[0],col[1],col[2],1)
        #return cVector(int(color[1], 16) / 0xf, int(color[2], 16) / 0xf, int(color[3], 16) / 0xf, 1)

    match = re.match(r"^rgba\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9.eE]+)\s*\)$", color)
    if match:
        return cVector(int(match[1])/255, int(match[2])/255, int(match[3])/255, float(match[4]))

    match = re.match(r"^rgb\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)$", color)
    if match:
        col = colors.hex2color(color)
        return cVector(col[0]/255,col[1]/255,col[2]/255,1)
        #return cVector(int(match[1])/255, int(match[2])/255, int(match[3])/255, 1)

    match = re.match(r"^rgb\s*\(\s*([0-9]+)%\s*,\s*([0-9]+)%\s*,\s*([0-9]+)%\s*\)$", color)
    if match:
        #col = colors.hex2color(color)
        #print (match[3])
        return cVector(int(match[1])/100, int(match[2])/100, int(match[3])/100,1)

        #return cVector(col[0]/100,col[1]/100,col[2]/100,1)
        #return cVector(int(match[1])/100, int(match[2])/100, int(match[3])/100, 1)

    match = re.match(r"^rgb\s*\(\s*([0-9]+)%\s*,\s*([0-9]+)%\s*,\s*([0-9]+)%\s*,\s*([0-9.eE]+)\s*\)$", color)
    if match:
        col = colors.hex2color(color)
        return cVector(col[0]/100,col[1]/100,col[2]/100,float(match[4]))
        #return cVector(int(match[1])/100, int(match[2])/100, int(match[3])/100, float(match[4]))
    
    match = re.match(r'rgb\((?:(\d{1,3}.?(?:\d{1,50}\%)?)(?:\,?)(\d{1,3}.?(?:\d{1,50}\%)?)(?:\,?)(\d{1,3}.?(?:\d{1,50}\%)?)(?:))\)', color)
    if match:
        match = re.findall('\d*\.?\d+',str(match))
        #print ((int(match[2])/100)*255,(float(match[3])/100)*255,(float(match[4])/100)*255,1)
        r = (float(match[2])/100)*255
        g = (float(match[3])/100)*255
        b = (float(match[4])/100)*255
        #print (int(r)/255,int(g)/255,int(b)/255)
        return cVector(r/255,g/255,b/255,1)

    if color == "transparent":
        return cVector(0, 0, 0, 0)

    match = re.match(r"^hsl\s*\(\s*([0-9]+)\s*,\s*([0-9]+)%\s*,\s*([0-9]+)%\s*\)$", color)
    if match:
        return cVector(*(hsl_to_rgb(int(match[1])/360, int(match[2])/100, int(match[3])/100) + [1]))

    match = re.match(r"^hsl\s*\(\s*([0-9]+)\s*,\s*([0-9]+)%\s*,\s*([0-9]+)%\s*,\s*([0-9.eE]+)\s*\)$", color)
    if match:
        return cVector(*(hsl_to_rgb(int(match[1])/360, int(match[2])/100, int(match[3])/100) + [float(match[4])]))

    if color in {"currentColor", "inherit"}:
        return current_color

    return cVector(*color_table[color])




