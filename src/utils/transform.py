import math
import numpy as np

from .vector import NVector

def cVector(*args):
    param =[]
    for elem in args :
        param.append (elem)
    p = np.array(param)
    v = NVector(p)
    l = len(p)
    list1 = v[0].tolist()

    l = v[0].tolist()

    ##print ("fucking vec type : ", type(l))
    ##print ("return vec is :", l)

    return l

def _sign(x):
    if x < 0:
        return -1
    return 1


class TransformMatrix:
    scalar = float

    def __init__(self):
        """ Creates an Identity matrix """
        self.to_identity()

    def clone(self):
        m = TransformMatrix()
        m._mat = list(self._mat)
        return m

        return self

    def __getitem__(self, key):
        row, col = key
        return self._mat[row*4+col]

    def __setitem__(self, key, value):
        row, col = key
        self._mat[row*4+col] = self.scalar(value)

    @property
    def a(self):
        return self[0, 0]

    @a.setter
    def a(self, v):
        self[0, 0] = self.scalar(v)

    @property
    def b(self):
        return self[0, 1]

    @b.setter
    def b(self, v):
        self[0, 1] = self.scalar(v)

    @property
    def c(self):
        return self[1, 0]

    @c.setter
    def c(self, v):
        self[1, 0] = self.scalar(v)

    @property
    def d(self):
        return self[1, 1]

    @d.setter
    def d(self, v):
        self[1, 1] = self.scalar(v)

    @property
    def tx(self):
        return self[3, 0]

    @tx.setter
    def tx(self, v):
        self[3, 0] = self.scalar(v)

    @property
    def ty(self):
        return self[3, 1]

    @ty.setter
    def ty(self, v):
        self[3, 1] = self.scalar(v)

    def __str__(self):
        return str(self._mat)

    def scale(self, x, y=None):
        if y is None:
            y = x

        m = TransformMatrix()
        m.a = x
        m.d = y
        self *= m
        return self

    def translate(self, x, y=None):
        if y is None:
            x, y = x
        m = TransformMatrix()
        m.tx = x
        m.ty = y
        self *= m
        return self

    def skew(self, x_rad, y_rad):
        m = TransformMatrix()
        m.c = math.tan(x_rad)
        m.b = math.tan(y_rad)
        self *= m
        return self

    #def skew_from_axis(self, ax, angle):
    def skew_from_axis(self, skew, axis):
        self.rotate(axis)
        #self.rotate(angle)
        m = TransformMatrix()
        #m.c = math.tan(ax)
        m.c = math.tan(skew)
        self *= m
        #self.rotate(-angle)
        self.rotate(-axis)
        return self

    def row(self, i):
        return NVector(self[i, 0], self[i, 1], self[i, 2], self[i, 3])

    def column(self, i):
        return NVector(self[0, i], self[1, i], self[2, i], self[3, i])

    def to_identity(self):
        self._mat = [
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ]

    def apply(self, vector):
        vector3 = NVector(vector[0], vector[1], 0, 1)
        r = NVector(
            self.column(0).dot(vector3),
            self.column(1).dot(vector3),
        )
        #print ("r :",r)
        return ([r[0],r[1]])

    @classmethod
    def rotation(cls, radians):
        m = cls()
        m.a = math.cos(radians)
        m.b = -math.sin(radians)
        m.c = math.sin(radians)
        m.d = math.cos(radians)

        return m

    def __mul__(self, other):
        m = TransformMatrix()
        for row in range(4):
            for col in range(4):
                m[row, col] = self.row(row).dot(other.column(col))
        return m

    def __imul__(self, other):
        m = self * other
        self._mat = m._mat
        return self

    def rotate(self, radians):
        self *= TransformMatrix.rotation(radians)
        return self

    def extract_transform(self):
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        tx = self.tx
        ty = self.ty

        dest_trans = {
            "translation": cVector(tx, ty),
            "angle": 0,
            "scale": cVector(1, 1),
            "skew_axis": 0,
            "skew_angle": 0,
        }

        delta = a * d - b * c
        if a != 0 or b != 0:
            r = math.hypot(a, b)
            dest_trans["angle"] = - _sign(b) * math.acos(a/r)
            sx = r
            sy = delta / r
            dest_trans["skew_axis"] = 0
            sm = 1
        else:
            r = math.hypot(c, d)
            dest_trans["angle"] = math.pi / 2 + _sign(d) * math.acos(c / r)
            sx = delta / r
            sy = r
            dest_trans["skew_axis"] = math.pi / 2
            sm = -1

        dest_trans["scale"] = NVector(sx, sy)

        skew = sm * math.atan2(a * c + b * d, r * r)
        dest_trans["skew_angle"] = skew

        return dest_trans

    def to_css_2d(self):
        return "matrix(%s, %s, %s, %s, %s, %s)" % (
            self.a, self.b, self.c, self.d, self.tx, self.ty
        )

