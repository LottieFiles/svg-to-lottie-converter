from typing import List, Union, Any
#from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field , Schema


from . import properties, helpers

MDProperty = Union[properties.MultiDimensional, properties.MultiDimensionalKeyframed]
Value = Union[properties.Value, properties.ValueKeyframed]


class TextJustify(Enum):
    Left = 0
    Right = 1
    Center = 2

class TextMoreOptions(BaseModel):
    alignment: properties.MultiDimensional = Field(None, alias='a', description='Text Grouping Alignment')
    apGrouping: helpers.TextGrouping = Field(helpers.TextGrouping.Characters, alias='g', description='Text Anchor Point Grouping')


class MaskedPath(BaseModel):
    """ ref : https://github.com/airbnb/lottie-web/blob/master/player/js/utils/text/TextAnimatorProperty.js """
    """ could not find in schema """
    # have no effing idea what these means, f = first, l = last , r = ??? 
    f: Value = Field(None)
    l: Value = Field(None)
    r: float = Field(None)
    mask: float = Field(None, alias='m')


class TextAnimatorDataProperty(BaseModel):
    rotation : Value = Field(None,alias='r',description='Text animator Rotation')
    rotateX : Value = Field(None,alias='rx',description='Text animator Rotation X')
    rotateY : Value = Field(None,alias='ry',description='Text animator Rotation Y')
    skew : Value = Field(None,alias='sk',description='Text animator Skew')
    skewAxis : Value = Field(None,alias='sa',description='Text animator Skew Axis')
    scale: properties.MultiDimensional = Field(None,alias='s',description='Text animator Scale')
    anchor : properties.MultiDimensional = Field(None,alias='a',description='Text animator Anchor Point')
    opacity : Value = Field(None,alias='o',description='Text animator Opacity')
    position : properties.MultiDimensional = Field(None,alias='p',description='Text animator Position')
    strokeWidth : Value = Field(None,alias='sw',description='Text animator Stroke Width')
    strokeColor : properties.MultiDimensional = Field(None,alias='sc',description='Text animator Stroke Color')
    fillColor : properties.MultiDimensional = Field(None,alias='fc',description='Text animator Fill Color')
    fillHue : Value = Field(None,alias='fh',description='Text animator Fill Hue')
    fillSaturation : Value = Field(None,alias='fs',description='Text animator Fill Saturation')
    fillBrightness : Value = Field(None,alias='fb',description='Text animator Fill Brightness')
    tracking : Value = Field(None,alias='t',description='Text animator Tracking')


#@dataclass
class TextDocument(BaseModel):
    fontFamily : str  = Field(None,alias='f', description='font family')
    fillColorData : MDProperty = Field(None, alias='fc', description='font color')
    fontSize : float = Field(None, alias='s', description='font size')
    lineHeight : float = Field(None, alias='lh', description='line height')
    textFrameSize : MDProperty = Field(None, alias='sz', description='wrap size')
    text : str = Field(None, alias='t', description='text')
    justification : TextJustify = Field(None, alias='j', description='justify')
    strokeColorData : MDProperty = Field(None, alias='sc')
    strokeWidth : float = Field(None, alias='sw')
    strokeOverFill : bool = Field(None, alias='of')
    textFramePosition : MDProperty = Field(None, alias='ps')
    tracking : int = Field(None, alias='tr')


#@dataclass
class TextDataKeyframe(BaseModel):
    start : TextDocument = Field(None, alias='s', description='start vale of keyframe segment')
    time : float = Field(None, alias='t', description='start time of keyframe segment')


#@dataclass
class TextData(BaseModel):
    keyframes : TextDataKeyframe = Field(None, alias='k', description='keyframes')


#@dataclass
class TextAnimatorData(BaseModel):
    properties : TextAnimatorDataProperty  = Field(None, alias='a', description='properties')
    data : TextData = Field(None, alias='d',description='Text Data')
    moreOptions : TextMoreOptions = Field(None, alias='m',description='Text More Options')
    maskedPath : MaskedPath = Field(None, alias='p',description='')


#@dataclass
class Font(BaseModel):
    ascent : float = Field(None, alias='ascent')
    fontFamily : str = Field(None, alias='fFamily')
    name : str = Field(None, alias='fName')
    fontStyle : str = Field(None, alias='fStyle')

#@dataclass
class FontList(BaseModel):
    list: Font = Field(None, alias='list')
