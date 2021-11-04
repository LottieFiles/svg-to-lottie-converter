#from PIL import Image
from io import BytesIO
import base64
import os

from typing import List, Union

from pydantic import BaseModel, Field , Schema

from . import layers
from . import shapes
from . import properties

#AnyLayer = Union[layers.ShapeLayer, layers.SolidLayer, layers.PreCompLayer, layers.ImageLayer, layers.NullLayer, layers.TextLayer]
AnyLayer = Union[layers.PreCompLayer, layers.NullLayer, layers.ImageLayer,
                 layers.ShapeLayer, layers.SolidLayer, layers.TextLayer]

#AnyLayer = layers.ShapeLayer # Union[layers.ShapeLayer, layers.PreCompLayer]

class Precomp(BaseModel):
    referenceID: str = Field(...,alias='id') # id pointing to the source composition defined on 'assets' object
    #timeRemapping: float = Field(..., alias='tm') # Comp's Time remapping
    #type : int = Field(helpers.LayerType.Precomposition.value,alias='ty')

    layers: List[AnyLayer] = Field(...,alias='layers')

    class Config:
        allow_population_by_field_name = True


class Image(BaseModel):
    height: float = Field(..., alias='h') 
    width: float = Field(..., alias='w')
    id: str = Field(..., alias='id')
    image: str = Field(..., alias='p')
    imagePath: str = Field(..., alias='u')
    embedded: int = Field(..., alias='e')

    def load(self, file, format="png"):
    
        im = Image.open(file)
        self.width, self.height = im.size
        output = BytesIO()
        im.save(output, format=format)
        self.image = "data:image/%s;base64,%s" % (
            format,
            base64.b64encode(output.getvalue()).decode("ascii")
        )
        self.embedded = True
        if not self.id_:
            if isinstance(file, str):
                self.id_ = os.path.basename(file)
            elif hasattr(file, "name"):
                self.id_ = os.path.basename(file.name)
            else:
                self.id_ = "image_%s" % id(self)
        return self


class characterData(BaseModel):
    shape: properties.Shape = Field(None, alias='shape')

class Chars(BaseModel):
    character: str = Field(...,alias='ch')
    font_family : str = Field(...,alias='fFamily')
    font_size : float = Field(...,alias='size')
    font_style : str = Field(...,alias='style')
    width: float = Field(...,alias='w')
    data: characterData = Field(...,alias='data')

    class Config:
        allow_population_by_field_name = True

class Fonts:
    def __init__(self):
        raise NotImplementedError()


class Markers:
    def __init__(self):
        raise NotImplementedError()


class Meta(BaseModel):
    generator : str = Field('LottieFiles pytoolkit 0.1',alias='g')
    authorName: str = Field(None,alias='a')
    keywords: str = Field(None,alias='k')
    description: str = Field(None,alias='d')
    backgroundColor: str = Field(None,alias='tc')

    class Config:
        allow_population_by_field_name = True


