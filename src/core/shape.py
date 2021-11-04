import contextlib
import contextlib
import functools
import math
import contextlib

import numpy as np
import triangle as tr


from model import bezier
from utils.vector import NVector as Vector
from utils.vector import NVector #as Vector

from model.bezier import BezierPoint
from model.properties import Value, MultiDimensional,OffsetKeyframe,MDBezier,ShapeProp
from model.bezier import Bezier
from model import *

def lottieVector(*args):
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

def Point(x,y):
    return (Value(value=[x,y]))

def Size(x,y):
    return (Value(value=[x,y]))

__all__ = ['PShape']

def _ensure_editable(func):
    """A decorater that ensures that a shape is in 'edit' mode.
    """
    @functools.wraps(func)
    def editable_method(instance, *args, **kwargs):
        if not instance._in_edit_mode:
            raise ValueError('{} only works in edit mode'.format(func.__name__))
        return func(instance, *args, **kwargs)
    return editable_method

def _apply_transform(func):
    """Apply the matrix transformation to the shape.
    """
    @functools.wraps(func)
    def mfunc(instance, *args, **kwargs):
        tmat = func(instance, *args, **kwargs)
        instance._matrix = instance._matrix.dot(tmat)
        return tmat
    return mfunc

def _call_on_children(func):
    """Call the method on all child shapes
    """
    @functools.wraps(func)
    def rfunc(instance, *args, **kwargs):
        rval = func(instance, *args, **kwargs)
        for child in instance.children:
            rfunc(child, *args, **kwargs)
        return rval
    return rfunc

class PShape:
    def __init__(self, vertices=[], fill_color='auto',
                 stroke_color='auto', stroke_weight="auto", 
                 stroke_join="auto", stroke_cap="auto", 
                 visible=False, attribs='closed',
                 children=None, contour=None):
        # basic properties of the shape
        self._vertices = np.array([])
        self.contour = contour or np.array([])
        self._edges = None
        self._outline = None
        self._outline_vertices = None

        self.attribs = set(attribs.lower().split())
        self._fill = None
        self._stroke = None
        self._stroke_weight = None
        self._stroke_cap = None
        self._stroke_join = None

        self._matrix = np.identity(4)
        self._transform_matrix = np.identity(4)
        self._transformed_draw_vertices = None

        # a flag to check if the shape is being edited right now.
        self._in_edit_mode = False
        self._vertex_cache = None

        # The triangulation used to render the shapes.
        self._tri = None
        self._tri_required = not ('point' in self.attribs) and \
                             not ('path' in self.attribs)
        self._tri_vertices = None
        self._tri_edges = None
        self._tri_faces = None

        if len(vertices) > 0:
            self.vertices = vertices

        self.fill = fill_color
        self.stroke = stroke_color
        self.stroke_weight = stroke_weight
        self.stroke_cap = stroke_cap
        self.stroke_join = stroke_join

        self.children = children or []
        self.visible = visible

    def add_child(self, child):
        """Add a child shape to the current shape
        :param child: Child to be added
        :type child: PShape
        """
        self.children.append(child)

    def transform_matrix(self, mat):
        self._transform_matrix = mat
    
    @_call_on_children
    def apply_transform_matrix(self, mat):
        self._matrix = self._matrix.dot(mat)

class Lottie:
    def setup(width=512,height=512, endFrame=60, frameRate=60):
        an = animation.Animation()
        an.width = width
        an.height = height
        an.endFrame = endFrame
        an.frameRate = frameRate
        return an
    
    #@property
    def addShapeLayer(object,name=None):
        layer = layers.ShapeLayer()
        object.add_layer(layer)
        layer.name = name

        return layer
    
    #@property
    def addGroup(object,name=None):
        ##print  ("group add")
        group = object.add_shape(layers.Group())
        group.name = name

        return group
    
    def addPath(object,name=None,path=None,posx=None,posy=None):

        pshape = object.add_shape(shapes.Path())
        pshape.shape.value = (path)

        if (posx is not None) and (posy is not None):
            transform = object.add_shape(shapes.Transform())
            transform.position = Value(value=[posx,posy])
        
        return pshape

    def Color(object,fillColor=None,fillOpacity=None,strokeColor=None,strokeOpacity=None,strokeWidth=None, *args):
        if fillColor is not None:
            fill = object.add_shape(shapes.Fill())
            fill.color = Value(value=fillColor)

            if fillOpacity is not None:
                fill.opacity = Value(value=fillOpacity)
            else:
                fill.opacity = Value(value=100)

        
        if strokeColor is not None:
            stroke = object.add_shape(shapes.Stroke())
            stroke.color = Value(value=strokeColor)

            if strokeOpacity is not None:
                stroke.opacity = Value(value=strokeOpacity)
            else:
                stroke.opacity = Value(value=100)
            
            if strokeWidth is not None:
                stroke.width = Value(value=strokeWidth)
            else:
                stroke.width = Value(value=1)
        else:
            stroke = object.add_shape(shapes.Stroke())
            stroke.color = Value(value=[0, 0, 0, 1])
            stroke.opacity = Value(value=100)
            if strokeWidth is not None:
                stroke.width = Value(value=strokeWidth)
            else:
                stroke.width = Value(value=1)

    def drawEllipse(object, x,y,w,h,fillColor=None,fillOpacity=None,strokeColor=None,strokeOpacity=None,strokeWidth=None, *args):
        ellipse = object.add_shape(shapes.Ellipse())
        ellipse.position = Value(value=[x,y])
        ellipse.size = Value(value=[w, h])
        
        if fillColor is not None:
            fill = object.add_shape(shapes.Fill())
            fill.color = Value(value=fillColor)

            if fillOpacity is not None:
                fill.opacity = Value(value=fillOpacity)
            else:
                fill.opacity = Value(value=100)

        
        if strokeColor is not None:
            stroke = object.add_shape(shapes.Stroke())
            stroke.color = Value(value=strokeColor)

            if strokeOpacity is not None:
                stroke.opacity = Value(value=strokeOpacity)
            else:
                stroke.opacity = Value(value=100)
            
            if strokeWidth is not None:
                stroke.width = Value(value=strokeWidth)
            else:
                stroke.width = Value(value=1)
        else:
            stroke = object.add_shape(shapes.Stroke())
            stroke.color = Value(value=[0, 0, 0, 1])
            stroke.opacity = Value(value=100)
            stroke.width = Value(value=1)

        object.add_shape(shapes.Transform())

    def drawRect(object, x,y,w,h,fillColor=None,fillOpacity=None,strokeColor=None,strokeOpacity=None,strokeWidth=None, *args):
        rect = object.add_shape(shapes.Rect())
        rect.position = Value(value=[x,y])
        rect.size = Value(value=[w, h])
        
        if fillColor is not None:
            fill = object.add_shape(shapes.Fill())
            fill.color = Value(value=fillColor)

            if fillOpacity is not None:
                fill.opacity = Value(value=fillOpacity)
            else:
                fill.opacity = Value(value=100)

        
        if strokeColor is not None:
            stroke = object.add_shape(shapes.Stroke())
            stroke.color = Value(value=strokeColor)

            if strokeOpacity is not None:
                stroke.opacity = Value(value=strokeOpacity)
            else:
                stroke.opacity = Value(value=100)
            
            if strokeWidth is not None:
                stroke.width = Value(value=strokeWidth)
            else:
                stroke.width = Value(value=1)
        else:
            stroke = object.add_shape(shapes.Stroke())
            stroke.color = Value(value=[0, 0, 0, 1])
            stroke.opacity = Value(value=100)
            stroke.width = Value(value=1)

        object.add_shape(shapes.Transform())

    def drawStar(object, x,y,innerRadius=10,outerRadius=10,fillColor=None,fillOpacity=None,strokeColor=None,strokeOpacity=None,strokeWidth=None, *args):
        rect = object.add_shape(shapes.Star())
        rect.position = Value(value=[x,y])
        #rect.size = Value(value=[w, h])
        rect.innerRadius = Value(value=innerRadius)
        rect.outerRadius = Value(value=outerRadius)

        if fillColor is not None:
            fill = object.add_shape(shapes.Fill())
            fill.color = Value(value=fillColor)

            if fillOpacity is not None:
                fill.opacity = Value(value=fillOpacity)
            else:
                fill.opacity = Value(value=100)

        
        if strokeColor is not None:
            stroke = object.add_shape(shapes.Stroke())
            stroke.color = Value(value=strokeColor)

            if strokeOpacity is not None:
                stroke.opacity = Value(value=strokeOpacity)
            else:
                stroke.opacity = Value(value=100)
            
            if strokeWidth is not None:
                stroke.width = Value(value=strokeWidth)
            else:
                stroke.width = Value(value=1)
        else:
            stroke = object.add_shape(shapes.Stroke())
            stroke.color = Value(value=[0, 0, 0, 1])
            stroke.opacity = Value(value=100)
            stroke.width = Value(value=1)

        object.add_shape(shapes.Transform())

class Ellipse:
    def __init__(self, center, radii, xrot):
        """
        \param center      2D vector, center of the ellipse
        \param radii       2D vector, x/y radius of the ellipse
        \param xrot        Angle between the main axis of the ellipse and the x axis (in radians)
        """
        self.center = center
        self.radii = radii
        self.xrot = xrot

    def point(self, t):
        return Vector(
            self.center[0]
            + self.radii[0] * math.cos(self.xrot) * math.cos(t)
            - self.radii[1] * math.sin(self.xrot) * math.sin(t),

            self.center[1]
            + self.radii[0] * math.sin(self.xrot) * math.cos(t)
            + self.radii[1] * math.cos(self.xrot) * math.sin(t)
        )

    def derivative(self, t):
        return Vector(
            - self.radii[0] * math.cos(self.xrot) * math.sin(t)
            - self.radii[1] * math.sin(self.xrot) * math.cos(t),

            - self.radii[0] * math.sin(self.xrot) * math.sin(t)
            + self.radii[1] * math.cos(self.xrot) * math.cos(t)
        )

    def to_bezier(self, anglestart, angle_delta):
        #print ("----- to_bezier ------")
        points = []
        angle1 = anglestart
        angle_left = abs(angle_delta)
        step = math.pi / 2
        sign = -1 if anglestart+angle_delta < angle1 else 1

        # We need to fix the first handle
        firststep = min(angle_left, step) * sign
        alpha = self._alpha(firststep)

        #print ("self.derivative(angle1) : ",self.derivative(angle1))
        #print ("alpha : ",alpha)
        #print ("self.point(angle1) type: ", type (self.point(angle1)))

        selfderivativeangle1 = NVector(self.derivative(angle1)[0],self.derivative(angle1)[1])
        q1 = selfderivativeangle1 * alpha

        #print ("lottieVector(self.point(angle1)) :" , (self.point(angle1)))
        #print ("lottieVector(0, 0) :",lottieVector(0, 0))
        #print ("q1 :",q1)

        points.append(BezierPoint([self.point(angle1)[0],self.point(angle1)[1]], lottieVector(0, 0), q1))

        # Then we iterate until the angle has been completed
        tolerance = step / 2
        while angle_left > tolerance:
            lstep = min(angle_left, step)
            step_sign = lstep * sign
            angle2 = angle1 + step_sign
            angle_left -= abs(lstep)

            alpha = self._alpha(step_sign)
            p2 = self.point(angle2)

            selfderivativeangle2 = NVector(self.derivative(angle2)[0],self.derivative(angle2)[1])

            q2 = (selfderivativeangle2 * alpha)
            qq2 = [q2[0],q2[1]]

            #print ("p2 :",type(p2))
            #print ("qq2 : ",qq2)
            #print ("qq2 :",type(qq2))
            negqq2 = [-x for x in qq2]

            #print ("(BezierPoint(p2, -q2, q2)) :",(p2, negqq2, qq2))
            points.append(BezierPoint(p2, negqq2, qq2))
            angle1 = angle2
        
        #print ("points : ",points)
        #print ("points type : ",type(points))
        return points

    def _alpha(self, step):
        return math.sin(step) * (math.sqrt(4+3*math.tan(step/2)**2) - 1) / 3

    @classmethod
    def from_svg_arc(cls, start, rx, ry, xrot, large, sweep, dest):
        """
        #print ("---- from_svg_arc types---")
        #print (type(start),start)
        #print (type(rx),rx)
        #print (type(ry),ry)
        #print (type(xrot),xrot)
        #print (type(large),large)
        #print (type(sweep),dest)
        #print ("---------------")
        """
        start = NVector(start[0],start[1])
        dest = NVector(dest[0],dest[1])

        rx = abs(rx)
        ry = abs(ry)

        x1 = start[0]
        y1 = start[1]
        x2 = dest[0]
        y2 = dest[1]
        phi = math.pi * xrot / 180

        x1p, y1p = _matrix_mul(phi, (start-dest)/2, -1)

        cr = x1p ** 2 / rx**2 + y1p**2 / ry**2
        if cr > 1:
            s = math.sqrt(cr)
            rx *= s
            ry *= s

        dq = rx**2 * y1p**2 + ry**2 * x1p**2
        pq = (rx**2 * ry**2 - dq) / dq
        cpm = math.sqrt(max(0, pq))
        if large == sweep:
            cpm = -cpm
        cp = NVector(cpm * rx * y1p / ry, -cpm * ry * x1p / rx)

        #print ("type _matrix_mul(phi, cp) : ",type(_matrix_mul(phi, cp)))
        cc = (_matrix_mul(phi, cp))
        #print (cc)
        c = NVector(cc[0],cc[1]) + NVector((x1+x2)/2, (y1+y2)/2)
        #print ("c = ",c)
        theta1 = _angle(NVector(1, 0), NVector((x1p - cp[0]) / rx, (y1p - cp[1]) / ry))
        deltatheta = _angle(
            NVector((x1p - cp[0]) / rx, (y1p - cp[1]) / ry),
            NVector((-x1p - cp[0]) / rx, (-y1p - cp[1]) / ry)
        ) % (2*math.pi)

        if not sweep and deltatheta > 0:
            deltatheta -= 2*math.pi
        elif sweep and deltatheta < 0:
            deltatheta += 2*math.pi

        #print (cls(c, lottieVector(rx, ry), phi))
        #print ("rx, ry : ",rx,ry)
        #print ("type rx,ry : ",type(rx),type(ry))
        #print ("theta1 : ",theta1)
        #print (type(theta1))
        #print ("deltatheta : ",deltatheta)
        #print ("type :", type(deltatheta))

        return cls(c, lottieVector(rx, ry), phi), theta1, deltatheta


def _matrix_mul(phi, p, sin_mul=1):
    c = math.cos(phi)
    s = math.sin(phi) * sin_mul

    xr = c * p.x - s * p.y
    yr = s * p.x + c * p.y
    return Vector(xr, yr)


def _angle(u, v):
    arg = math.acos(max(-1, min(1, u.dot(v) / (u.length * v.length))))
    if u[0] * v[1] - u[1] * v[0] < 0:
        return -arg
    return arg
    

