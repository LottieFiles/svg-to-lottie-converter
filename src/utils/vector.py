# Origianl parser and related helper functions code from https://gitlab.com/mattbas/python-lottie
# SVG parse using https://gitlab.com/mattbas/python-lottie/. 
# Change to original code : Generating Lottie using pydantic based object model.


import operator
import math
import pathlib

import json
import pydantic.json 
from pydantic import BaseModel, Field , Schema
from typing import List, Dict


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        #print ("here.....")
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def vop(op, a, b):
    return list(map(op, a, b))


class NVector():

    def __init__(self, *components):
        self.components = list(components)

    def __str__(self):
        return str(self.components)

    def __repr__(self):
        #print (json.dumps(self.components))
        #return json.dumps(self.components)
        #print (json.dumps(self.components))
        #return json.dumps(self.components)
        return "<NVector %s>" % self


    def __len__(self):
        return len(self.components)

    def to_list(self):
        return list(self.components)

    def __add__(self, other):
        return NVector(*vop(operator.add, self.components, other.components))

    def __sub__(self, other):
        return NVector(*vop(operator.sub, self.components, other.components))

    def __mul__(self, scalar):
        if isinstance(scalar, NVector):
            return NVector(*vop(operator.mul, self.components, scalar.components))
        return NVector(*(c * scalar for c in self.components))

    def __truediv__(self, scalar):
        return NVector(*(c / scalar for c in self.components))

    def __iadd__(self, other):
        self.components = vop(operator.add, self.components, other.components)
        return self

    def __isub__(self, other):
        self.components = vop(operator.sub, self.components, other.components)
        return self

    def __imul__(self, scalar):
        if isinstance(scalar, NVector):
            self.components = vop(operator.mul, self.components, scalar.components)
        else:
            self.components = [c * scalar for c in self.components]
        return self

    def __itruediv__(self, scalar):
        self.components = [c / scalar for c in self.components]
        return self

    def __neg__(self):
        return NVector(*(-c for c in self.components))

    def __getitem__(self, key):
        return self.components[key]

    def __setitem__(self, key, value):
        self.components[key] = value

    def __eq__(self, other):
        return self.components == other.components

    def __abs__(self):
        return NVector(*(abs(c) for c in self.components))

    @property
    def length(self):
        return math.sqrt(sum(map(lambda x: x**2, self.components)))

    def dot(self, other):
        return sum(map(operator.mul, self.components, other.components))

    def clone(self):
        return NVector(*self.components)

    def lerp(self, other, t):
        return self * (1-t) + other * t

    @property
    def x(self):
        return self.components[0]

    @x.setter
    def x(self, v):
        self.components[0] = v

    @property
    def y(self):
        return self.components[1]

    @y.setter
    def y(self, v):
        self.components[1] = v

    @property
    def z(self):
        return self.components[2]

    @z.setter
    def z(self, v):
        self.components[2] = v

    def element_scaled(self, other):
        return NVector(*vop(operator.mul, self.components, other.components))

    def cross(self, other):
        """
        \pre len(self) == len(other) == 3
        """
        a = self
        b = other
        return NVector(
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @property
    def polar_angle(self):
        """
        \pre len(self) == 2
        """
        return math.atan2(self.y, self.x)


def Point(x, y):
    return NVector(x, y)


def Size(x, y):
    return NVector(x, y)


def Point3D(x, y, z):
    return NVector(x, y, z)


def Color(r, g, b):
    return NVector(r, g, b)


def PolarNVector(length, theta):
    return NVector(length * math.cos(theta), length * math.sin(theta))
