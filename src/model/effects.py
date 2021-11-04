from __future__ import annotations
from typing import List, Union ,Any, Optional,Type,TypeVar
from enum import Enum

from pydantic import BaseModel, Field , Schema
from . import properties, helpers
#from . helpereffect import Group
# this is gonna suck
Value = Union[int,properties.Value, properties.ValueKeyframed]
MDProperty = Union[properties.MultiDimensional, properties.MultiDimensionalKeyframed]
    
class EffectValue(BaseModel):
    index: int = Field (None,alias='ix')
    matchName: str = Field(None,alias='mn')
    name: str = Field(None,alias='nm')
    type: int = Field(None,alias='ty')


class EffectType(Enum):
    dropDown = 7,
    fill = 21,
    group = 5,
    layer = 0,
    point = 2,
    proLevel = 23,
    slider = 0,
    stroke = 22,
    tint = 20,
    tritone = 23,
    color = 2,
    angle = 1,
    checkBox = 7


class BaseEffect(BaseModel):
    #type: List [EffectType] = Field(...,alias='ty')
    index: int = Field (None,alias='ix')
    name: str = Field(None,alias='nm')
    matchName: str = Field(None,alias='mn')
    class Config:
        allow_population_by_field_name = True


class Angle(BaseEffect):
    type: int = Field(EffectType.angle.value,alias='ty')
    value : Value = Field(..., alias='v', description='Effect value.')
    class Config:
        allow_population_by_field_name = True


class Color(BaseEffect):
    type: int = Field(EffectType.color.value,alias='ty')
    value : MDProperty = Field(..., alias='v', description='Effect value.')

    class Config:
        allow_population_by_field_name = True


class Point(BaseEffect):
    type: int = Field(EffectType.point.value,alias='ty')
    value : MDProperty = Field(..., alias='v', description='Effect value.')

    class Config:
        allow_population_by_field_name = True    


class Layer(BaseEffect):
    type: int = Field(EffectType.layer.value,alias='ty')
    value : MDProperty = Field(..., alias='v', description='Effect value.')

    class Config:
        allow_population_by_field_name = True


class checkBox(BaseEffect):
    type: int = Field(EffectType.checkBox.value,alias='ty')
    value : Value = Field(..., alias='v', description='Effect value.')

    class Config:
        allow_population_by_field_name = True


class Slider(BaseEffect):
    type: int = Field(EffectType.slider.value,alias='ty')
    value : Value = Field(..., alias='v', description='Effect value.')  

    class Config:
        allow_population_by_field_name = True


class dropDown(BaseEffect):
    type: int = Field(EffectType.dropDown.value,alias='ty')
    value : Value = Field(..., alias='v', description='Effect value.')

    class Config:
        allow_population_by_field_name = True


class FillValues(BaseModel):
    point: Point
    dropdown0: dropDown
    color: Color
    dropdown1: dropDown
    slider0: Slider
    slider1: Slider
    slider2: Slider


class Fill(BaseEffect):
    type: int = Field(EffectType.fill.value,alias='ty')
    effect: FillValues = Field(None, alias='ef', description='Effect List of properties')

    class Config:
        allow_population_by_field_name = True


class noValue(BaseModel):
    pass


class customValue(BaseModel):
    pass


class proLevelsValues(BaseModel):
    dropdown: dropDown
    novalue0 : noValue
    novalue1 : noValue
    slider0: Slider
    slider1: Slider
    slider2: Slider
    slider3: Slider
    slider4: Slider
    novalue2 : noValue
    slider5: Slider
    slider6: Slider
    slider7: Slider
    slider8: Slider
    slider9: Slider
    novalue3 : noValue
    slider10: Slider
    slider11: Slider
    slider12: Slider
    slider13: Slider
    slider14: Slider
    novalue4 : noValue
    slider15: Slider
    slider16: Slider
    slider17: Slider
    slider18: Slider
    slider19: Slider
    novalue5 : noValue
    slider20: Slider
    slider21: Slider
    slider22: Slider
    slider23: Slider
    slider24: Slider

class proLevels(BaseEffect):
    type: int = Field(EffectType.proLevel.value,alias='ty')
    effect: proLevelsValues = Field(None, alias='ef', description='Effect List of properties')

    class Config:
        allow_population_by_field_name = True


class StrokeValues(BaseModel):
    color0: Color
    checkbox0: checkBox
    checkbox1: checkBox
    color1: Color
    slider0: Slider
    slider1: Slider
    slider2: Slider
    slider3: Slider
    slider4: Slider
    dropdown0: dropDown
    dropdown1: dropDown


class Stroke(BaseEffect):
    type: int = Field(EffectType.stroke.value,alias='ty')
    effect: StrokeValues = Field(None, alias='ef', description='Effect List of properties')

    class Config:
        allow_population_by_field_name = True


class TintValues(BaseModel):
    color0: Color
    color1: Color
    Slider: Slider


class Tint(BaseEffect):
    type: int = Field(EffectType.tint.value,alias='ty')
    effect: TintValues = Field(None, alias='ef', description='Effect List of properties')

    class Config:
        allow_population_by_field_name = True


class TritoneValues(BaseModel):
    color0: Color
    color1: Color
    color2: Color
    Slider: Slider


class Tritone(BaseEffect):
    type: int = Field(EffectType.tritone.value,alias='ty')
    effect: TintValues = Field(None, alias='ef', description='Effect List of properties')

    class Config:
        allow_population_by_field_name = True

#gIndex = Union[dropDown,Slider,Angle,Color,Point,checkBox,noValue,customValue,Layer,Tint,Fill,Stroke,Tritone,proLevels]
#gGroup = Union[gIndex,Group]
#gIndex = List[Union[dropDown,Slider,Angle,Color,Point,checkBox,noValue,customValue,Layer,Tint,Fill,Stroke,Tritone,proLevels]]
#Index = Union[Group,dropDown,Slider,Angle,Color,Point,checkBox,noValue,customValue,Layer,Tint,Fill,Stroke,Tritone,proLevels]

Groups = TypeVar('Group')
gIndex = Union[Groups,dropDown,Slider,Angle,Color,Point,checkBox,noValue,
                customValue,Layer,Tint,Fill,Stroke,Tritone,proLevels]

class Group(BaseModel):
    type: int = Field(...,alias='ty')
    name: str = Field(None,alias='nm')
    numberOfProperties: int = Field(None,alias='np') 
    matchName: str = Field(None,alias='mn')
    index: int = Field (None,alias='ix')
    enabled: int = Field(None,alias='en')
    effect: List[gIndex] = Field(..., alias='ef', description='Effect List of properties')

    class Config:
        allow_population_by_field_name = True

Group.update_forward_refs()

Index = Union[Group,dropDown,Slider,Angle,Color,Point,checkBox,noValue,customValue,
              Layer,Tint,Fill,Stroke,Tritone,proLevels]

class Effect(BaseModel):
    type: Optional[int] = Field(None,alias='ty')
    index: int = Field (None,alias='ix')
    name: str = Field(None,alias='nm')
    matchName: str = Field(None,alias='mn')
    effects : List[Index] = Field(...,alias='ef')

    enabled: Optional[int] = Field(None,alias='en')
    numberOfProperties: Optional[int] = Field(None,alias='np') 

    class Config:
        allow_population_by_field_name = True

Effect.update_forward_refs()

