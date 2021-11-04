# Refer to https://gitlab.com/mattbas/python-lottie

from typing import List, Union, Any, Optional
from dataclasses import  field
from pydantic import BaseModel, Field , Schema , ValidationError, validator

from utils.vector import NVector as Vector
from utils.vector import NVector #as Vector

from model import *
import inspect


## \ingroup Lottie
class KeyframeBezierHandle(BaseModel):
    x: float = Field(0)  # X axis
    y: float = Field(0)  # Y axis

    class Config:
        allow_population_by_field_name = True

class Linear(KeyframeBezierHandle):
    """!
    Linear easing, the value will change from start to end in a straight line
    """
    def __call__(self, keyframe):
        keyframe.outValue = KeyframeBezierHandle(
            0,
            0
        )
        keyframe.inValue = KeyframeBezierHandle(
            1,
            1
        )


class EaseIn:
    """!
    The value lingers near the start before accelerating towards the end
    """
    def __init__(self, delay=1/3):
        self.delay = delay

    def __call__(self, keyframe):
        keyframe.out_value = KeyframeBezierHandle(
            self.delay,
            0
        )
        keyframe.in_value = KeyframeBezierHandle(
            1,
            1
        )


class EaseOut:
    """!
    The value starts fast before decelerating towards the end
    """
    def __init__(self, delay=1/3):
        self.delay = delay

    def __call__(self, keyframe):
        keyframe.out_value = KeyframeBezierHandle(
            0,
            0
        )
        keyframe.in_value = KeyframeBezierHandle(
            1-self.delay,
            1
        )


class Jump:
    """!
    Jumps to the end value at the end of the keyframe
    """
    def __call__(self, keyframe):
        keyframe.jump = True


class Sigmoid:
    """!
    Combines the effects of EaseIn and EaseOut
    """
    def __init__(self, delay=1/3):
        self.delay = delay

    def __call__(self, keyframe):
        keyframe.out_value = KeyframeBezierHandle(
            self.delay,
            0
        )
        keyframe.in_value = KeyframeBezierHandle(
            1 - self.delay,
            1
        )
