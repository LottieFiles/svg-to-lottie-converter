# Refer to https://gitlab.com/mattbas/python-lottie

from typing import List, Union, Any, Optional
from dataclasses import  field
from pydantic import BaseModel, Field , Schema , ValidationError, validator
import inspect
import copy
import math
from functools import reduce
import numpy as np
from utils.vector import NVector #as Vector
from . import easing

#from .shapes import Ellipse


#from pydantic.dataclasses import dataclass


def Vector(*args):
    param =[]
    for elem in args :
        param.append (elem)
    p = np.array(param)
    v = NVector(p)
    l = len(p)
    list1 = v[0].tolist()

    l = v[0].tolist()

    ###print ("fucking vec type : ", type(l))
    ###print ("return vec is :", l)

    return l

Number = Union[float, int]

class Vertices(BaseModel):
    x: Number = Field(None)  # X axis
    y: Number = Field(None)  # Y axis

BezierCurveVertices = List[Number]

class Bezier(BaseModel):
    x: Optional[float] = Field(None)  # X axis
    y: Optional[float] = Field(None)  # Y axis
    #x: Number = Field(None) # X axis
    #y: Number = Field(None) # Y axis

    
class MDBezier(BaseModel):
    x: List[Number] = Field(None) # X axis
    y: List[Number] = Field(None) # Y axis

LottieBezier = Union[Bezier, MDBezier]


class Keyframe(BaseModel):
    time: float  = Field(None, alias='t', description='Start time of keyframe segment.')
    inValue: MDBezier = Field(None, alias='i', description='Bezier curve interpolation in value.') 
    outValue: MDBezier = Field(None, alias='o', description='Bezier curve interpolation out value.') 

    class Config:
        allow_population_by_field_name = True

class OffsetKeyframe(BaseModel):
    start: List[Number] = Field(None, alias='s', description='Start value')
    time: Number  = Field(None, alias='t', description='Start time')
    inValue: LottieBezier = Field(None, alias='i', description='In value for Bezier curve')
    outValue: LottieBezier = Field(None, alias='o', description= 'Out value for Bezier curve')
    #inValue: Bezier = Field(default=field(default_factory=list), alias='i', description='In value for Bezier curve')
    inTan: List[float] = Field(None,alias='ti')
    outTan: List[float] = Field(None,alias='to')


    class Config:
        allow_population_by_field_name = True

#SomeType = Optional[Union[Number, List[Optional[Union[int,OffsetKeyframe, None]]]]]

ValueTypeMD = Union[Number,OffsetKeyframe]
ValueType = Union[Number,List[Number]]


class Value(BaseModel):
    value: ValueType = Field(None, alias='k', description='Value or keyframe') # can be value or keyframe ??
    expression: Optional[str] = Field(default=None, alias='x', description='Expression')
    propertyIndex: Optional[int] = Field(None, alias='ix', description='Property Index') #default=0
    animated: int = Field(None, alias='a', description='animated or not?') #animated

    #value: Number = Field(None, alias='k', description='Value') # can be value or keyframe ??
    #keyframes: OffsetKeyframe = Field(None,alias='k', description='keyframed value')
    #value: ValueType = Field(None, alias='k', description='Value or keyframe') # can be value or keyframe ??


    class Config:
        allow_population_by_field_name = True
        orm_mode = True
    
    #def clone(self):
    #    return self.__class__()
    
    def clone(self):
        c = copy.deepcopy(super())
        #c._index_gen._i = self._index_gen._i
        return c
    
    def get_value(self, object, time=0):
        #print ("value: ",self.value)
        #print ("type value : ", type(self.value))

        #val = self.value[0].start
        #print ("v: ",v)
        return self.value

    def new_get_value(self, time=0):
        if not self.animated == None:
            return self.value

        #if not self.keyframes:
        #    return None

        #v = self._get_value_helper(time)[0]

        if self.animated == 1 and self.keyframes:
            return v[0]
        
        return v



     
class ValueKeyframe(BaseModel):
    start: List[Number] = Field(None, alias='s', description='Start')
    time: Number  = Field(None, alias='t', description='Time')
    inValue: MDBezier  = Field(None, alias='i', description='Bezier in')
    outValue: MDBezier  = Field(None, alias='o', description='Bezier out')

    class Config:
        allow_population_by_field_name = True


class ValueKeyframed(BaseModel):
    animated: int = Field(default=1, alias='a', description='animated')
    value: List[ValueKeyframe] = Field(None, alias='k', description='Value Keyframes')
    propertyIndex: int = Field(None, alias='ix', description='Property Index')
    expression: str  = Field(None, alias='x', description='Expression')

    class Config:
        allow_population_by_field_name = True

PosValue = Union[Value, ValueKeyframed]

class PositionContainer(BaseModel):
    split: bool = Field(...,alias='s')
    positionX: PosValue  = Field(..., alias='x', description='x')
    positionY: PosValue  = Field(..., alias='y', description='y')


class MultiDimensional(BaseModel):
    value: List[ValueTypeMD] = Field(None, alias='k', description='Value')  # Value 
    expression: Optional[str] = Field(default=None, alias='x', description='Expression')
    propertyIndex: Optional[int] = Field(None, alias='ix', description='Property Index') #default=0
    animated: int = Field(None, alias='a', description='not animated')

    class Config:
        allow_population_by_field_name = True
    
    def get_value(self, time=0):
        
        """!
        @brief Returns the value of the property at the given frame/time
        """
        #print ("==> get_value in MultiDimensional" )
        if not self.animated:
            return self.value

        if not self.keyframes:
            return None

        val = self.keyframes[0].start
        for i in range(len(self.keyframes)):
            k = self.keyframes[i]
            if time - k.time <= 0:
                if k.start is not None:
                    val = k.start

                kp = self.keyframes[i-1] if i > 0 else None
                if kp:
                    t = (time - kp.time) / (k.time - kp.time)
                    end = kp.end
                    if end is None:
                        end = val
                    if end is not None:
                        val = kp.interpolated_value(t, end)
                    return val, end, kp, t
                return val, None, None, None
            if k.end is not None:
                val = k.end
        return val, None, None, None




class MultiDimensionalKeyframed(BaseModel):
    animated: int = Field(None, alias='a', description='not animated')
    value: List[OffsetKeyframe] = Field(None, alias='k',description='Property Value keyframes')
    expression: str  = Field(None, alias='x', description='Property Expression. An AE expression that modifies the value.')
    propertyIndex: int = Field(None, alias='ix', description='Property Index. Used for expressions')
    inTangent: List[Number]  = Field (None, alias='ti', description='In Spatial Tangent. Only for spatial properties. Array of numbers.') 
    outTangent: List[Number]  = Field (None, alias='to', description='Out Spatial Tangent. Only for spatial properties. Array of numbers.')

    class Config:
        allow_population_by_field_name = True

    


class pathBezier(BaseModel):
    inPoint: List[LottieBezier]  = Field([],alias='i', description='Bezier curve In points. Array of 2 dimensional arrays.')
    outPoint: List[LottieBezier] = Field([], alias='o', description='Bezier curve Out points. Array of 2 dimensional arrays.')
    vertices: List[LottieBezier] = Field([], alias='v', description='Bezier curve Vertices. Array of 2 dimensional arrays.')
    closed: bool = Field(False, alias='c', description='Closed property of shape') 

    class Config:
        allow_population_by_field_name = True
    
    def clone(self):
        clone = Bezier()
        clone.closed = self.closed
        clone.inPoint = [p.clone() for p in self.inPoint]
        clone.outPoint = [p.clone() for p in self.outPoint]
        clone.vertices = [p.clone() for p in self.vertices]
        #clone.rel_tangents = self.rel_tangents
        return clone

    def insert_point(self, index, pos, inp=(0, 0), outp=(0, 0)):
        #print ("--- insert_point ----")
        ###print (type(pos))
        ###print ("inp :",(inp))
        ###print ("outp :",outp)

        try:
            v = pos.value
        except:
            v = pos
        
        try:
            i = inp.value 
        except:
            i = inp

        try:
            o = outp.value 
        except:
            o = outp

        self.vertices.insert(index, v)
        self.inPoint.insert(index,i ) 
        self.outPoint.insert(index,o ) 

        #print (type(v))
        #print (type(i))
        #print (type(o))

        return self

    def add_point(self, pos, inp=(0, 0), outp=(0, 0)):

        if isinstance(pos,NVector):
            pos = [pos[0],pos[1]]

        if isinstance(inp,NVector):
            inp = [inp[0],inp[1]]
        
        if isinstance(outp,NVector):
            outp = [outp[0],outp[1]]

        if isinstance(pos,np.ndarray):
            pos = pos.tolist()

        if isinstance(inp,np.ndarray):
            inp = inp.tolist()

        
        if isinstance(outp,np.ndarray):
            outp = outp.tolist()

        self.insert_point(len(self.vertices), pos, inp, outp)
        return self
    
    def add_smooth_point(self, pos, inp):
        neginp = copy.deepcopy(inp)
        neginp.value[:] = [-1*x for x in neginp.value]

        self.add_point(pos, inp, neginp)
        return self

    def close(self, closed=True):
        """!
        Updates self.closed
        \returns \c self, for easy chaining
        """
        self.closed = closed
        return self


class ShapeProp(BaseModel):
    value: pathBezier  = Field(..., alias='k', description='Property Value')
    expression: str  = Field(None, alias='x', description='Property Expression. An AE expression that modifies the value.')
    propertyIndex: int = Field(None, alias='ix', description='Property Index. Used for expressions.')
    animated: int = Field(None, alias='a', description='Defines if property is animated')

    class Config:
        allow_population_by_field_name = True
    
    def get_value(self, time=0):
        """!
        @brief Returns the value of the property at the given frame/time
        """
        if not self.animated:
            return self.value

        if not self.keyframes:
            return None

        val = self.keyframes[0].start
        for i in range(len(self.keyframes)):
            k = self.keyframes[i]
            if time - k.time <= 0:
                if k.start is not None:
                    val = k.start

                kp = self.keyframes[i-1] if i > 0 else None
                if kp:
                    end = kp.end
                    if end is None:
                        end = val
                    if end is not None:
                        val = kp.interpolated_value((time - kp.time) / (k.time - kp.time), end)
                break
            if k.end is not None:
                val = k.end
        return val



class ShapePropKeyframe(BaseModel):
    start: List[ShapeProp]  = Field (None, alias='s', description='Start value of keyframe segment.')
    time: float  = Field(None, alias='t', description='Start time of keyframe segment.')
    inValue: LottieBezier = Field(None, alias='i', description='Bezier curve interpolation in value.') 
    outValue: LottieBezier = Field(None, alias='o', description='Bezier curve interpolation out value.') 

    class Config:
        allow_population_by_field_name = True


class Shape(BaseModel):
    value: ShapeProp  = Field(..., alias='k', description='Property Value')
    expression: str  = Field(None, alias='x', description='Property Expression. An AE expression that modifies the value.')
    propertyIndex: int = Field(None, alias='ix', description='Property Index. Used for expressions.')
    animated: int = Field(None, alias='a', description='Defines if property is animated')

    class Config:
        allow_population_by_field_name = True


class ShapeKeyframed(BaseModel):
    value: List[ShapePropKeyframe]  = Field(None, alias='k',description='')
    expression: str  = Field(None, alias='x', description='Property Expression. An AE expression that modifies the value.')
    propertyIndex: int = Field(None, alias='ix', description='Property Index. Used for expressions.')
    inTangent: List[Number]  = Field(None, alias='ti', description='In Spatial Tangent. Only for spatial properties. Array of numbers.')
    outTangent: List[Number]  = Field(None, alias='to', description='Out Spatial Tangent. Only for spatial properties. Array of numbers.')
    animated: int = Field(default=1, alias='a', description='')

    class Config:
        allow_population_by_field_name = True


class GradientColors(BaseModel):
    count: int = Field(0,alias='p',description='Count')
    colors: MultiDimensional = Field(MultiDimensional(value=()),alias='k',description='Represents colors and offsets in a gradient')

    class Config:
        allow_population_by_field_name = True
    
    @validator('colors')
    def set_the_colors(cls,v):
        if v:
            self.set_colors(colors)
        return v

    #def set_colors(self, colors, keyframe=None):
    def set_colors(self, colors, keyframe=None, alpha=None):

        #flat = self._flatten_colors(colors)
        flat = self._flatten_colors(colors, alpha)
        if self.colors.animated and keyframe is not None:
            if keyframe > 1:
                #self.colors.keyframes[keyframe-1].end = flat
                self._add_to_flattened(offset, color, self.colors.keyframes[keyframe-1].end.components)
            #self.colors.keyframes[keyframe].start = flat
            self._add_to_flattened(offset, color, self.colors.keyframes[keyframe].start.components)
        else:
            #self.colors.clear_animation(flat)
            self._add_to_flattened(offset, color, self.colors.value.components)
        self.count = len(colors)

    #def _flatten_colors(self, colors):
    #    return NVector(*reduce(
    #        lambda a, b: a + b,
    #        map(
    #            lambda it: [it[0] / (len(colors)-1)] + it[1].components,
    #            enumerate(colors),
    #        )
    #    ))
    def _flatten_colors(self, colors, alpha):
        if alpha is None:
            alpha = any(len(c) > 3 for c in colors)

        def offset(n):
            return [n / (len(colors)-1)]

        flattened_colors = NVector(*reduce(
            lambda a, b: a + b,
            map(
                lambda it: offset(it[0]) + it[1].components[:3],
                enumerate(colors),
            )
        ))
        if alpha:
            flattened_colors.components += reduce(
                lambda a, b: a + b,
                map(
                    lambda it: offset(it[0]) + [self._get_alpha(it[1])],
                    enumerate(colors),
                )
            )
        return flattened_colors

    def _get_alpha(self, color):
        if len(color) > 3:
            return color[3]
        return 1

    def _add_to_flattened(self, offset, color, flattened):
        flat = [offset] + color[:3]
        rgb_size = 4 * self.count

        if len(flattened) == rgb_size:
            # No alpha
            flattened.extend(flat)
            if self.count == 0 and len(color) > 3:
                flattened.append(offset)
                flattened.append(color[3])
        else:
            flattened[rgb_size:rgb_size] = flat
            flattened.append(offset)
            flattened.append(self._get_alpha(color))



    def add_color(self, offset, color, keyframe=None):
        #print ("self.colors.value.components @ start :",self.colors.value)
        #flat = [offset] + color
        if self.colors.animated:
            if keyframe is None:
                for kf in self.colors.keyframes:
                    if kf.start:
                        kf.start += flat
                    if kf.end:
                        kf.end += flat
            else:
                if keyframe > 1:
                    #self.colors.keyframes[keyframe-1].end += flat
                    self._add_to_flattened(offset, color, self.colors.keyframes[keyframe-1].end.value)
                #self.colors.keyframes[keyframe].start += flat
                self._add_to_flattened(offset, color, self.colors.value)
        else:
            #self.colors.value += flat
            self._add_to_flattened(offset, color, self.colors.value)
        self.count += 1

    #def add_keyframe(self, time, colors=None, ease=easing.Linear()):
    #    self.colors.add_keyframe(time, self._flatten_colors(colors) if colors else NVector(), ease)
    def add_keyframe(self, time, colors=None, ease=easing.Linear(), alpha=None):
        self.colors.add_keyframe(time, self._flatten_colors(colors, alpha) if colors else NVector(), ease)

    def get_sane_colors(self, keyframe=None):
        if keyframe is not None:
            colors = self.colors.keyframes[keyframe].start
        else:
            colors = self.colors.value
        return self._sane_colors_from_flat(colors)

    def _sane_colors_from_flat(self, colors):
        if len(colors) == 4 * self.count:
            for i in range(self.count):
                off = i * 4
                yield colors[off], NVector(*colors[off+1:off+4])
        else:
            for i in range(self.count):
                off = i * 4
                aoff = self.count * 4 + i * 2 + 1
                yield colors[off], NVector(colors[off+1], colors[off+2], colors[off+3], colors[aoff])

    def sane_colors_at(self, time):
        return self._sane_colors_from_flat(self.colors.get_value(time))




