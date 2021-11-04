from typing import List, Union, Any , Optional, TypeVar
#from dataclasses import dataclass
from enum import Enum
import math
import json
import copy
import numpy as np

from pydantic import BaseModel, Field , Schema
from . import properties, helpers
from .properties import Value as pValue, MultiDimensional,ShapeProp
#from .bezier import Bezier
from core.base import Index
from utils.vector import NVector 
from utils.vector import Point,Size
#from utils.vector import Vector
#import properties, helpers

MDProperty = Union[properties.MultiDimensional, properties.MultiDimensionalKeyframed]
Value = Union[properties.Value, properties.ValueKeyframed]

def traverse(o, tree_types=(list, tuple)):
    if isinstance(o, tree_types):
        for value in o:
            for subvalue in traverse(value, tree_types):
                yield subvalue
    else:
        yield o

def normalizelist(val):
    if not ((type(val) == int) or (type(val) == float)):
        if not (val == None):
            val = (val.tolist())
            num = (list(traverse(val)))
            return num[0]
        else:
            return 0
    else:
        return val

def Vector(*args):
    param =[]
    for elem in args :
        param.append (elem)
    p = np.array(param)
    v = NVector(p)
    l = len(p)
    list1 = v[0].tolist()

    l = v[0]#.tolist()

    ##print ("fucking vec type : ", type(l))

    return l

class BoundingBox:
    """!
    Shape bounding box
    """
    def __init__(self, x1=None, y1=None, x2=None, y2=None):
        #print ("this -> ",normalizelist(x1),normalizelist(y1),normalizelist(x2),normalizelist(y2))

        self.x1 = normalizelist(x1)
        self.y1 = normalizelist(y1)
        self.x2 = normalizelist(x2)
        self.y2 = normalizelist(y2)

    def include(self, x, y):
        """!
        Expands the box to include the point at x, y
        """
        if x is not None:
            if self.x1 is None or self.x1 > x:
                self.x1 = x
            if self.x2 is None or self.x2 < x:
                self.x2 = x
        if y is not None:
            if self.y1 is None or self.y1 > y:
                self.y1 = y
            if self.y2 is None or self.y2 < y:
                self.y2 = y

    def expand(self, other):
        """!
        Expands the bounding box to include another bounding box
        """
        self.include(other.x1, other.y1)
        self.include(other.x2, other.y2)

    def center(self):
        """!
        Center point of the bounding box
        """
        return Vector((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def isnull(self):
        """!
        Whether the box is default-initialized
        """
        return self.x1 is None or self.y2 is None

    def __repr__(self):
        return "<BoundingBox [%s, %s] - [%s, %s]>" % (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self):
        if self.isnull():
            return 0
        return self.x2 - self.x1

    @property
    def height(self):
        if self.isnull():
            return 0
        return self.y2 - self.y1

    #@property
    def size(self):
        return (Vector(self.width), Vector(self.height))
    
    def cords(self):
        #print (self.x1, self.y1, self.x2, self.y2)
        return (self.x1, self.y1, self.x2, self.y2)



class ShapeElement(BaseModel):
    type: str = Field(None,alias='ty')
    hidden: bool = Field(default=False, alias='hd', description='')
    name: str = Field(default=None, alias='nm')
    matchName: str = Field(None,alias='mn')
    propertyIndex: Optional[int] = Field(default=None,alias='ix')  # Property Index
    index: Optional[int]  = Field(None, alias='ind', description='layer index')

    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    class Config:
        allow_population_by_field_name = True  
    
    def bounding_box(self, time=0):
        return BoundingBox()


class Shape(BaseModel):
    type: str = Field(None,alias='ty')
    hidden: bool = Field(None, alias='hd', description='')
    name: str = Field(None, alias='nm')
    matchName: str = Field(None,alias='mn')
    propertyIndex: int = Field(None,alias='ix')  # Property Index
    index: int  = Field(None, alias='ind', description='layer index')
    direction: float = Field(None,alias='d')   # After Effect's Direction. Direction how the shape is drawn. Used for trim path   
    #path: Union[properties.Shape, properties.ShapeKeyframed] = Field(...,alias='ks', description='Shapes vertices')
    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    autoOrient: Optional[int] = Field(None,alias="ao")


    class Config:
        allow_population_by_field_name = True


class Path(Shape):
    type: str = Field("sh",alias='ty')
    shape :ShapeProp  = Field(ShapeProp(value=[]),alias="ks")

    class Config:
        allow_population_by_field_name = True
    
    def bounding_box(self, time=0):
        """!
        Bounding box of the shape element at the given time
        """
        pos = self.shape.value

        bb = BoundingBox()
        for v in pos.vertices:
            bb.include(*v)

        return bb


        #return BoundingBox()

class Ellipse(BaseModel): #_EllipseBase
    type : str = Field("el", alias='ty', description='Shape content type.')  
    name: str = Field(None, alias='nm')
    matchName: str = Field(None,alias='mn')
    hidden: bool = Field(None, alias='hd', description='')
    #propertyIndex: int = Field(None,alias='ix')  # Property Index
    #index: int  = Field(None, alias='ind', description='layer index')
    direction: int = Field(None,alias='d')   # After Effect's Direction. Direction how the shape is drawn. Used for trim path

    position: MDProperty = Field(MultiDimensional(value=[50,50]), alias='p', description='Ellipses position')
    size: MDProperty  = Field(MultiDimensional(value = [100,100]), alias='s', description='Ellipses size')
    
    td: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    class Config:
        allow_population_by_field_name = True
    

    def bounding_box(self, time=0):
        ###print (type(self.position))
        pos = self.position.value #get_value(self,time)
        sz = self.size.value #get_value(time)

        return BoundingBox(
            pos[0] - sz[0]/2,
            pos[1] - sz[1]/2,
            pos[0] + sz[0]/2,
            pos[1] + sz[1]/2,
        )

    


class Fill(BaseModel):
    type : str = Field(default='fl', alias='ty', description='Shape content type.')
    name: str = Field(default=None, alias='nm')
    matchName: str = Field(None,alias='mn')    
    opacity: Value  = Field(None, alias='o', description='Fill Opacity')
    color: MDProperty  = Field(None, alias='c', description='Fill Color')
    r: int = Field(None,alias='r') # WTF is "r" ???
    blendMode: Optional[int] = Field(None,alias='bm',description='Blend Mode')
    hidden: bool = Field(None, alias='hd', description='')

    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    class Config:
        allow_population_by_field_name = True
    
    def bounding_box(self, time=0):
        """!
        Bounding box of the shape element at the given time
        """
        return BoundingBox()



class GradientType(Enum):
    Linear = 1
    Radial = 2


class Gradient(BaseModel):
    type : str = Field(" ", alias='ty')
    startPoint : MDProperty = Field(properties.MultiDimensional(value=(0,0)), alias='s', description='Gradient Start Point')
    endPoint : MDProperty = Field(properties.MultiDimensional(value=(0,0)), alias='e', description='Gradient End Point')
    gradientType : GradientType = Field(None, alias='t', description='Gradient Type')
    highlightLength : Value = Field(properties.Value(value=0), alias='h', description='Gradient Highlight Length. Only if type is Radial')
    highlightAngle : Value = Field(properties.Value(value=0), alias='a', description='Highlight Angle. Only if type is Radial')
    colors : properties.GradientColors = Field(properties.GradientColors(), alias='g', description='Gradient Colors')
    
    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    class Config:
        allow_population_by_field_name = True



class GFill(ShapeElement,Gradient):
    type : str = Field(default='gf', alias='ty')
    opacity : Value = Field(properties.Value(value=100),alias='o', description='Fill Opacity')

    class Config:
        allow_population_by_field_name = True

    def bounding_box(self, time=0):
        return BoundingBox()



class Stroke(BaseModel):
    type : str = Field(default='st', alias='ty', description='Shape content type.') 
    matchName: str = Field(None,alias='mn')
    name: str = Field(default=None, alias='nm')
    lineCap : helpers.LineCap  = Field(None, alias='lc', description='Stroke Line Cap')
    lineJoin : helpers.LineJoin = Field(None, alias='lj', description='Stroke Line Join')
    miterLimit : Optional[int]  = Field(None, alias='ml', description='Stroke Miter Limit. Only if Line Join is set to Miter.')
    opacity : Value  = Field(properties.Value(value=100), alias='o', description='Stroke Opacity') 
    width : Value  = Field(properties.Value(value=1), alias='w', description='Width') 
    color : MDProperty  = Field(properties.MultiDimensional(value=(0,0)),alias='c', description='Stroke Color')
    blendMode: Optional[helpers.BlendMode] = Field(None,alias='bm',description='Blend Mode')
    hidden: Optional[bool] = Field(default=False, alias='hd', description='')
    propertyIndex: Optional[int] = Field(default=None,alias='ix')  # Property Index

    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    class Config:
        allow_population_by_field_name = True

    def bounding_box(self, time=0):
        return BoundingBox()


class GStroke(BaseModel) :
    type : str = Field(default='gs', alias='ty',description='Shape content type.')
    matchName: str = Field(None,alias='mn')
    name: str = Field(default=None, alias='nm')
    opacity : Value  = Field(..., alias='o', description='Stroke Opacity') 
    startPoint : MDProperty = Field(None, alias='s', description='Gradient Start Point')
    endPoint : MDProperty = Field(None, alias='e', description='Gradient End Point')
    gradientType : GradientType = Field(None, alias='t', description='Gradient Type')
    highlightLength : Value = Field(None, alias='h', description='Gradient Highlight Length. Only if type is Radial')
    highlightAngle : Value = Field(None, alias='a', description='Highlight Angle. Only if type is Radial')
    width : Value  = Field(..., alias='w', description='Width') 
    lineCap : helpers.LineCap  = Field(..., alias='lc', description='Stroke Line Cap')
    lineJoin : helpers.LineJoin = Field(..., alias='lj', description='Stroke Line Join')
    blendMode: Optional[helpers.BlendMode] = Field(None,alias='bm',description='Blend Mode')
    colors : properties.GradientColors = Field(None, alias='g', description='Gradient Colors')

    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    class Config:
        allow_population_by_field_name = True


class Merge (ShapeElement):
    type : str = Field(None, alias='ty',description='Shape content type.')
    mergeMode: int  = Field(...,alias='mm', description='Merge Mode')

    class Config:
        allow_population_by_field_name = True


class Rect(BaseModel):
    type : str = Field(default='rc', alias='ty',description='Shape content type.')
    name: str = Field(None, alias='nm')
    matchName: str = Field(None,alias='mn')
    hidden: bool = Field(None, alias='hd', description='')
    direction: float = Field(None,alias='d')   # After Effect's Direction. Direction how the shape is drawn. Used for trim path  

    propertyIndex: Optional[int] = Field(None,alias='ix')  # Property Index
    #index: int  = Field(None, alias='ind', description='layer index')
    size: MDProperty  = Field(MultiDimensional(value=(0,0)),alias='s', description='Rects size')
    position: MDProperty  = Field(MultiDimensional(value=(0,0)),alias='p', description='Rects position')
    rounded: Value  = Field(pValue(value=0),alias='r', description='Rects rounded corners')

    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    class Config:
        allow_population_by_field_name = True

    def bounding_box(self, time=0):
        pos = self.position.get_value(time)
        sz = self.size.get_value(time)

        return BoundingBox(
            pos[0] - sz[0]/2,
            pos[1] - sz[1]/2,
            pos[0] + sz[0]/2,
            pos[1] + sz[1]/2,
        )

    def to_bezier(self):
        """!
        Returns a Shape corresponding to this rect
        """
        shape = Path()
        kft = set()
        if self.position.animated:
            kft |= set(kf.time for kf in self.position.keyframes)
        if self.size.animated:
            kft |= set(kf.time for kf in self.size.keyframes)
        if self.rounded.animated:
            kft |= set(kf.time for kf in self.rounded.keyframes)
        if not kft:
            shape.shape.value = self._bezier_t(0)
        else:
            for time in sorted(kft):
                shape.shape.add_keyframe(time, self._bezier_t(time))
        return shape



class Modifier(ShapeElement):
    pass

class Repeater(Modifier):
    type : str =  Field(default='rp', alias='ty',description='Shape content type.')
    hidden: bool = Field(default=False, alias='hd', description='')
    name: str = Field(default=None, alias='nm')
    matchName: str = Field(None,alias='mn')
    propertyIndex: Optional[int] = Field(default=None,alias='ix')  # Property Index

    copies: Value = Field(..., alias='c', description='Number of Copies')
    offset: Value = Field(..., alias='o', description='Offset of Copies')
    composite: helpers.Composite = Field(...,alias='m', description='Composite of copies')
    transform: helpers.Transform  = Field(...,alias='tr',description='Transform values for each repeater copy')

    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    class Config:
        allow_population_by_field_name = True


class Round(Modifier):
    type : str = Field(default='rd', alias='ty',description='Shape content type.')
    radius: Value  = Field(..., alias='r', description='Rounded Corner Radius')

    class Config:
        allow_population_by_field_name = True


class StarType(Enum):
    Star = 1
    Polygon = 2



class Star(BaseModel):
    type : str = Field(default='sr', alias='ty',description='Shape content type.')
    hidden: bool = Field(None, alias='hd', description='')
    name: str = Field(None, alias='nm')
    matchName: str = Field(None,alias='mn')
    propertyIndex: int = Field(None,alias='ix')  # Property Index
    index: int  = Field(None, alias='ind', description='layer index')
    direction: float = Field(None,alias='d')   # After Effect's Direction. Direction how the shape is drawn. Used for trim path  
    position : MDProperty = Field(MultiDimensional(value=(0,0)), alias='p', description='Stars position')
    innerRadius : Value = Field(pValue(value=0), alias='ir', description='Stars inner roundness. (Star only)')
    innerRoundness : Value = Field(pValue(value=0), alias='is', description='Stars inner roundness. (Star only)')
    outerRadius : Value = Field(pValue(value=0), alias='or', description='Stars outer radius.')
    outerRoundness: Value  = Field(pValue(value=0),alias='os', description='Stars outer roundness.') 
    rotation : Value  = Field(pValue(value=0), alias='r', description='Stars rotation.') 
    points : Value = Field(pValue(value=5),alias='pt', description='Stars number of points.')
    starType : StarType = Field(StarType.Star, alias='sy', description='Stars type. Polygon or Star.')

    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF

    class Config:
        allow_population_by_field_name = True


class Transform(ShapeElement,helpers.Transform):
    """  Group transform """
    type : str = Field('tr', alias='ty',description='Shape content type.')

    class Config:
        allow_population_by_field_name = True

    def bounding_box(self, time=0):
        """!
        Bounding box of the shape element at the given time
        """
        return BoundingBox()

class Trim(ShapeElement):
    type : str = Field(default='tm', alias='ty',description='Shape content type.')
    start: Value = Field(...,alias='s',description='Trim Start.')
    end: Value  = Field(...,alias='e',description='Trim End.')
    angle: Value = Field(...,alias='o', description='Trim Offset.')

    composite: Optional[helpers.Composite] = Field(None,alias='m', description='Composite of copies')


    class Config:
        allow_population_by_field_name = True

Groups = TypeVar('Group')

ShapeElements = Union[Groups, Merge, Fill, GStroke, Shape, Rect, Ellipse , Stroke , Star, Round, GFill , Trim, Repeater, Transform]


class Group(ShapeElement): 
    type : str = Field(default='gr', alias='ty', description='Type')  
    matchName: str = Field(None,alias='mn')
    name: str = Field(default=None, alias='nm')
    numberOfProperties : int = Field(None,alias='np', description='Group number of properties. Used for expressions.') 
    shapes : List[ShapeElements] = Field([Transform()], alias='it', description='Group list of items') #items name changes to shapes
    
    blendMode: Optional[helpers.BlendMode] = Field(None,alias='bm',description='Blend Mode')
    cix : Optional[int] = Field(None,alias='cix')
    propertyIndex: int = Field(None,alias='ix',description='Property Index')
    hidden: bool = Field(default=False, alias='hd', description='')

    #need to chk this with jaa
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    cp: Optional[bool] = Field(None,alias='cp') # same WTF
    
    #transform: helpers.Transform = Field(Transform(), alias='ks', description='transform')

    class Config:
        allow_population_by_field_name = True
    
    _index_gen = Index()

    def add_shape(self, shape):
        #print ("f index : ",(self._index_gen))
        #self.shapes.insert(next(self._index_gen), shape) #items to shapes changed
        self.shapes.insert(-1, shape)

        return shape

    def insert_shape(self, index, shape):
        self.items.insert(index, shape)
        return shape

    @property
    def transform(self):
        
        #print ("transform type:" , type(self.shapes[-1]))

        #return self.shapes[0]

        return self.shapes[-1]

    def bounding_box(self, time=0):
        # Refer to https://gitlab.com/mattbas/python-lottie

        bb = BoundingBox()

        for v in self.shapes:
            bb.expand(v.bounding_box(time))
        


        _s = np.array(self.transform.scale.get_value(time)) / 100
        #s = (_s / 100).tolist()
        _a = np.array(self.transform.anchorPoint.get_value(time))

        #print ("a : ", _a)
        pp = self.transform.position.get_value(time)
        p = [pp[-2],pp[-1]] 

        #_p = np.array(self.transform.position.get_value(time)) - _a
        _p = np.array(p) - _a
        _r = np.array(self.transform.rotation.get_value(time)) * math.pi / 180

        _sx = _s[0]
        _sy = _s[1]
        _px = _p[0]
        _py = _p[1]

        #s = self.transform.scale.get_value(time) / 100
        #a = self.transform.anchorPoint.get_value(time)
        #p = self.transform.position.get_value(time) - a
        #r = self.transform.rotation.get_value(time) * math.pi / 180

        if not bb.isnull():
            bb.x1 = bb.x1 * _sx + _px

            bb.y1 = bb.y1 * _sy + _py
            bb.x2 = bb.x2 * _sx + _px
            bb.y2 = bb.y2 * _sy + _py
            if _r:
                bbc = bb.center()
                #print ("size type: ",type(bb.size()))
                bbs = (np.array(bb.size()) / 2).tolist()

                bbc = NVector(bbc[0],bbc[1])
                bbs = NVector(bbs[0],bbs[1])

                relc = (np.array(bbc) - _a).tolist()
                relc = NVector(relc[0],relc[1])
                _r += relc.polar_angle
                _a = NVector(_a[0],_a[1])

                bbc = _a + NVector(math.cos(_r), math.sin(_r)) * relc.length
                
                #print ("bbc :",bbc)

                #bb = BoundingBox(bbc.x - bbs.x, bbc.y - bbs.y, bbc.x + bbs.x, bbc.y + bbs.y)
                bb = BoundingBox(Vector(bbc.x - bbs.x), Vector(bbc.y - bbs.y),Vector(bbc.x + bbs.x), Vector(bbc.y + bbs.y))

        return bb


    def clone(self):
        c = copy.deepcopy(super())
        c._index_gen._i = self._index_gen._i
        return c


AnyShape = Union[
    Repeater,Group, Merge, Fill, Shape, Rect, Ellipse, Star,
    GFill, GStroke, Stroke, Trim,
    Round
]