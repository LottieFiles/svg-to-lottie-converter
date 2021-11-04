from typing import List, Optional, Union
from enum import Enum
import numpy as np
import math,copy

from pydantic import BaseModel, Field , Schema
#from pydantic.dataclasses import dataclass

from . import helpers
from . import shapes
from . import text
from . import properties

from . import effects
from . import effects as Effects

from .shapes import  AnyShape, Group

Value = Union[properties.Value, properties.ValueKeyframed]

class BoundingBox:
    # Refer to https://gitlab.com/mattbas/python-lottie

    """!
    Shape bounding box
    """
    def __init__(self, x1=None, y1=None, x2=None, y2=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

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

    def size(self):
        return Vector(self.width, self.height)


class AdjustmentLayer:
    def __init__(self):
        raise NotImplementedError()

class AudioLayer:
    def __init__(self):
        raise NotImplementedError()

class CameraLayer:
    def __init__(self):
        raise NotImplementedError()

class GuideLayer:
    def __init__(self):
        raise NotImplementedError()

class ImagePlaceholderLayer:
    def __init__(self):
        raise NotImplementedError()

class ImageSequenceLayer:
    def __init__(self):
        raise NotImplementedError()

class LightLayer:
    def __init__(self):
        raise NotImplementedError()

class VideoLayer:
    def __init__(self):
        raise NotImplementedError()

class VideoPlaceholderLayer:
    def __init__(self):
        raise NotImplementedError()


class Layer(BaseModel):
    is3D: int = Field(None, alias='ddd', description='3D?')
    index: int  = Field(None, alias='ind', description='layer index')
    name: str  = Field(None, alias='nm', description='name')
    parent: Optional[int] = Field(None, alias='parent',description='parent layer')
    timeStretch: float = Field(None, alias='sr', description='time stretching')
    autoOrient: int = Field(default=0, alias='ao', description='auto-orient')
    startFrame: int = Field(None, alias='ip', description='in point (first frame)')
    endFrame: float  = Field(None,alias='op', description='out point (last frame)')
    startTime: float  = Field(0, alias='st', description='start time')
    blendMode: helpers.BlendMode = Field(None,alias='bm', description='blend mode') #default=helpers.BlendMode.Normal.value

    htmlClass: Optional[str] = Field(None, alias='cl', description='HTML class')
    htmlId: Optional[str] = Field(None, alias='ln', description='HTML ID')
    hasMask: Optional[bool] = Field(None, alias='hasMask', description='has a mask?')
    masksProperties: Optional[List[helpers.Mask]] = Field(None, alias='masksProperties', description='List of Masks')
    effects: Optional[List[Effects.Effect]] = Field(None,alias='ef', description='List of Effects') 
    hidden: Optional[bool] = Field(None, alias='hd', description='hidden')

    class Config:
        allow_population_by_field_name = True

    
class ShapeLayer(Layer): #_LayerDefaultBase,
    type: int = Field(helpers.LayerType.Shape.value,alias='ty')
    transform: helpers.Transform = Field(helpers.Transform(), alias='ks', description='transform')
    shapes: List[AnyShape] = Field([], alias='shapes', description='shape items')
    cp: Optional[bool] = Field(None,alias='cp') # WTF is cp ?
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is td??
    trackMatteType: Optional[int] = Field(None,alias='tt', description='Track Matte Type') # WTF is td??

    class Config:
        allow_population_by_field_name = True

    def add_shape(self, shape):
        self.shapes.append(shape)
        return shape

    def insert_shape(self, index, shape):
        self.shapes.insert(index, shape)
        return shape
    
    def bounding_box(self, time=0):
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


class SolidLayer(Layer):
    colorHex: str = Field(..., alias='sc') # Color of the solid in hex
    solidHeight: float = Field(..., alias='sh') # Height of the solid
    solidWidth: float = Field(..., alias='sw') #Width of the solid
    transform: helpers.Transform = Field(None, alias='ks', description='transform')

    type: int = Field(helpers.LayerType.Solid.value,alias='ty')

    class Config:
        allow_population_by_field_name = True


class ImageLayer(Layer):
    referenceID: str = Field(..., alias='refId') # id pointing to the source image defined on 'assets' object
    transform: helpers.Transform = Field(None, alias='ks', description='transform')
    cp: Optional[bool] = Field(None,alias='cp') # WTF is cp ?
    height: Optional[float] = Field(None,alias='h')
    width: Optional[float] = Field(None,alias='w')
    trackMatteType: Optional[int] = Field(None,alias='tt') # WTF is tt ?
    isTrackMatteType: Optional[int] = Field(None,alias='td') # WTF is tt ?


    type: int = Field(helpers.LayerType.Image.value,alias='ty')

    class Config:
        allow_population_by_field_name = True


class NullLayer(Layer):
    transform: helpers.Transform = Field(..., alias='ks', description='transform')
    shapes: Optional[List[AnyShape]] = Field(None, alias='shapes', description='shape items') #hack to test

    cp: Optional[bool] = Field(None,alias='cp') # this 3 is also a hack. need to chk
    isTrackMatteType: Optional[int] = Field(None,alias='td') 
    trackMatteType: Optional[int] = Field(None,alias='tt', description='Track Matte Type') 

    type : int = Field(helpers.LayerType.Null.value,alias='ty')

    class Config:
        allow_population_by_field_name = True


#  need to chk schema
class PreCompLayer(Layer):
    transform: helpers.Transform = Field(..., alias='ks', description='transform')
    referenceID: str = Field(..., alias='refId')  # id pointing to the source composition defined on 'assets' object
    timeRemapping: float = Field(None, alias='tm') # Comp's Time remapping
    type : int = Field(helpers.LayerType.Precomposition.value,alias='ty')
    height: int = Field(...,alias='h')
    width: int = Field(...,alias='w')

    class Config:
        allow_population_by_field_name = True


class TextLayer(Layer): 
    transform: helpers.Transform = Field(..., alias='ks', description='transform')
    type: int = Field(helpers.LayerType.Text.value,alias='ty') 
    t: List[text.TextAnimatorData] = Field(...,alias='t')



    