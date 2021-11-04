from typing import List, Union, Any
from enum import Enum
from pydantic import BaseModel, BaseConfig, Field , Schema
#from pydantic.dataclasses import dataclass

from . import layers
from . import assets 
from . assets import Meta, Chars
from core.base import Index
from .properties import *

#AnyLayer = Union[layers.NullLayer, layers.ImageLayer, layers.ShapeLayer, layers.SolidLayer, layers.PreCompLayer, layers.TextLayer]
AnyLayer = Union[layers.ShapeLayer, layers.SolidLayer, layers.PreCompLayer, layers.ImageLayer, layers.NullLayer, layers.TextLayer]

#AnyLayer = Union[layers.NullLayer, layers.ShapeLayer, layers.ImageLayer]
#AnyLayer = Union[layers.ShapeLayer, layers.NullLayer]

AnySource = Union[assets.Precomp, assets.Image, assets.Chars]
_version = "5.6.6"


#@dataclass
class Animation (BaseModel):
    version: str = Field(_version, alias='v', description='Bodymovin version')    
    startFrame: float = Field(0, alias='ip', description='in point (first frame)') 
    endFrame: float = Field(60, alias='op', description='out point (last frame)') 
    frameRate: float = Field(60, alias='fr', description='frame rate')
    width: float = Field(512, alias='w', description='width')
    is3D: int = Field(None,alias='ddd', description='is 3D')
    height: float = Field(512, alias='h', description='height')
    name: str = Field(None,alias='nm', description='composition name')  

    #assets: List[AnySource] = Field([],alias="assets") # Not used in SVG to Lottie
    layers: List[AnyLayer] = Field([],alias='layers')
    #chars: List[Chars] = Field([],alias='char') # Not used in SVG to Lottie
    #markers: List[Any] = Field([],alias='markers') # Not used in SVG to Lottie
    meta: Meta = Field(Meta(generator="LF SVG to Lottie"),alias='meta')
        
    class Config:
        allow_population_by_field_name = True
        orm_mode = True
    
    _index_gen = Index()


    
    def add_layer(self,layer):
        return self.insert_layer(len(self.layers), layer)

    def insert_layer(self, index, layer):
        self.layers.insert(index, layer)
        if layer.index is None:
            layer.index = next(self._index_gen)
        if layer.startFrame is None:
            layer.startFrame = self.startFrame
        if layer.endFrame is None:
            layer.endFrame = self.endFrame
        return layer

