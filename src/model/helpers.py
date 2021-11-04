from __future__ import annotations

from typing import Union, Optional, List
#from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field , Schema

from . import properties

MDProperty = Union[properties.PositionContainer, properties.MultiDimensionalKeyframed,properties.MultiDimensional]
Value = Union[properties.PositionContainer, properties.Value, properties.ValueKeyframed]
Shape = Union[properties.ShapeKeyframed, properties.Shape]

class TextGrouping(Enum):
    Characters = 1
    Word = 2
    Line = 3
    All = 4


class TextShape(Enum):
    Square = 1
    Ramp_Up = 2
    Ramp_Down = 3
    Triangle = 4
    Round = 5
    Smooth = 6


class BlendMode(Enum):
    Normal = 0
    Multiply = 1
    Screen = 2
    Overlay = 3
    Darken = 4
    Lighten = 5
    ColorDodge = 6
    ColorBurn = 7
    HardLight = 8
    SoftLight = 9
    Difference = 10
    Exclusion = 11
    Hue = 12
    Saturation = 13
    Color = 14
    Luminosity = 15


class LayerType(Enum):
    Precomposition = 0
    Solid = 1
    Image = 2
    Null = 3
    Shape = 4
    Text = 5
    Audio = 6
    Video_placeholder = 7
    Image_sequence = 8
    video = 9
    image_placholder = 10
    guide = 11
    adjustment = 12
    camera = 13
    light = 14


class TackMattType(Enum):
    NO_TRACK_MATTE = 1
    ALPHA = 2
    ALPHA_INVERTED = 3
    LUMA = 4
    LUMA_INVERTED = 5

class Composite(Enum):
    Above = 1
    Below = 2


class LineCap(Enum):
    Butt = 1
    Round = 2
    Square = 3


class LineJoin(Enum):
    Miter = 1
    Round = 2
    Bevel = 3


class MaskMode(Enum):
    None_ = 'n'
    Additive = 'a'
    Subtract = 's'
    Intersect = 'i'
    Lighten = 'l'
    Darken = 'd'
    Difference = 'f'


class Mask(BaseModel):
    inverted: bool = Field (None, alias='inv', description='Inverted') 
    name: str = Field(None, alias='nm', description='Name') 
    shape: Shape  = Field(None, alias='pt', description='Mask verts')
    opacity: Value  = Field(None,alias='o', description='Opacity')
    mode: MaskMode = Field(None,alias='mode')
    dilate : Value = Field(None,alias='x')

    class Config:
        allow_population_by_field_name = True


class Transform(BaseModel):
    """A transform"""
    type: str = Field("tr",alias='ty')
    opacity: Value  = Field(properties.Value(value=100), alias='o', description='Opacity')
    rotation: Value  = Field(properties.Value(value=0), alias='r', description='Rotation')
    position: MDProperty = Field(properties.MultiDimensional(value=(0,0)), alias='p', description='Position')
    anchorPoint: MDProperty = Field(properties.MultiDimensional(value=(0,0)), alias='a', description='Anchor Point')
    scale: MDProperty  = Field(properties.MultiDimensional(value=(100,100)), alias='s', description='Scale')
    
    autoOrient: Optional[int] = Field(None,alias="ao")

    xPositionTransform: Optional[Value] = Field(None,alias='px')  # Position X
    yPositionTransform: Optional[Value] = Field(None,alias='py')  # Position Y
    zPositionTransform: Optional[Value] = Field(None,alias='pz')  # Position Z

    xRotationTransform: Optional[Value] = Field(None,alias='rx')  # x rotation transform
    yRotationTransform: Optional[Value] = Field(None,alias='ry')  # y rotation transform
    zRotationTransform: Optional[Value] = Field(None,alias='rz')  # z rotation transform

    orientation: Optional[Value] = Field(None,alias='or')  # Position Z

    startOpacity: Optional[Value] = Field(None,alias='so')
    endOpacity: Optional[Value] = Field(None,alias='eo')


    skew: Optional[Value] = Field(properties.Value(value=0), alias='sk', description='Skew') #properties.Value(value=0)
    skewAxis: Optional[Value] = Field(properties.Value(value=0), alias='sa', description='Skew Axis') #properties.Value(value=0)

    #name: str = Field(None, alias='nm', description='Name') 

    class Config:
        allow_population_by_field_name = True
