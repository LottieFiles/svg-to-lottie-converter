# Origianl parser and related helper functions code from https://gitlab.com/mattbas/python-lottie
# SVG parse using https://gitlab.com/mattbas/python-lottie/. 
# Change to original code : Generating Lottie using pydantic based object model.

import colorsys
import math
from enum import Enum, auto

from .constants import colour_codes
from .vector import Vector, Color

def from_uint8(r,g,b):
    return Color(r,g,b) / 255

class ColorMode(Enum):
    RGB = auto()
    HSV = auto()
    HSL = auto()
    XYZ = auto()
    LUV = auto()
    LCH_uv = auto()
    LAB = auto()


class ParseColor:
    _conv_paths = {
        (ColorMode.RGB, ColorMode.RGB): [],
        (ColorMode.RGB, ColorMode.HSV): [],
        (ColorMode.RGB, ColorMode.HSL): [],
        (ColorMode.RGB, ColorMode.XYZ): [],
        (ColorMode.RGB, ColorMode.LUV): [ColorMode.XYZ],
        (ColorMode.RGB, ColorMode.LAB): [ColorMode.XYZ],
        (ColorMode.RGB, ColorMode.LCH_uv): [ColorMode.XYZ, ColorMode.LUV],

        (ColorMode.HSV, ColorMode.RGB): [],
        (ColorMode.HSV, ColorMode.HSV): [],
        (ColorMode.HSV, ColorMode.HSL): [],
        (ColorMode.HSV, ColorMode.XYZ): [ColorMode.RGB],
        (ColorMode.HSV, ColorMode.LUV): [ColorMode.RGB, ColorMode.XYZ],
        (ColorMode.HSV, ColorMode.LAB): [ColorMode.RGB, ColorMode.XYZ],
        (ColorMode.HSV, ColorMode.LCH_uv): [ColorMode.RGB, ColorMode.XYZ, ColorMode.LUV],

        (ColorMode.HSL, ColorMode.RGB): [],
        (ColorMode.HSL, ColorMode.HSV): [],
        (ColorMode.HSL, ColorMode.HSL): [],
        (ColorMode.HSL, ColorMode.XYZ): [ColorMode.RGB],
        (ColorMode.HSL, ColorMode.LUV): [ColorMode.RGB, ColorMode.XYZ],
        (ColorMode.HSL, ColorMode.LAB): [ColorMode.RGB, ColorMode.XYZ],
        (ColorMode.HSL, ColorMode.LCH_uv): [ColorMode.RGB, ColorMode.XYZ, ColorMode.LUV],

        (ColorMode.XYZ, ColorMode.RGB): [],
        (ColorMode.XYZ, ColorMode.HSV): [ColorMode.RGB],
        (ColorMode.XYZ, ColorMode.HSL): [ColorMode.RGB],
        (ColorMode.XYZ, ColorMode.XYZ): [],
        (ColorMode.XYZ, ColorMode.LUV): [],
        (ColorMode.XYZ, ColorMode.LAB): [],
        (ColorMode.XYZ, ColorMode.LCH_uv): [ColorMode.LUV],

        (ColorMode.LCH_uv, ColorMode.RGB): [ColorMode.LUV, ColorMode.XYZ],
        (ColorMode.LCH_uv, ColorMode.HSV): [ColorMode.LUV, ColorMode.XYZ, ColorMode.RGB],
        (ColorMode.LCH_uv, ColorMode.HSL): [ColorMode.LUV, ColorMode.XYZ, ColorMode.RGB],
        (ColorMode.LCH_uv, ColorMode.XYZ): [ColorMode.LUV],
        (ColorMode.LCH_uv, ColorMode.LUV): [],
        (ColorMode.LCH_uv, ColorMode.LAB): [ColorMode.LUV, ColorMode.XYZ],
        (ColorMode.LCH_uv, ColorMode.LCH_uv): [],

        (ColorMode.LUV, ColorMode.RGB): [ColorMode.XYZ],
        (ColorMode.LUV, ColorMode.HSV): [ColorMode.XYZ, ColorMode.RGB],
        (ColorMode.LUV, ColorMode.HSL): [ColorMode.XYZ, ColorMode.RGB],
        (ColorMode.LUV, ColorMode.XYZ): [],
        (ColorMode.LUV, ColorMode.LUV): [],
        (ColorMode.LUV, ColorMode.LAB): [ColorMode.XYZ],
        (ColorMode.LUV, ColorMode.LCH_uv): [],

        (ColorMode.LAB, ColorMode.RGB): [ColorMode.XYZ],
        (ColorMode.LAB, ColorMode.HSV): [ColorMode.XYZ, ColorMode.RGB],
        (ColorMode.LAB, ColorMode.HSL): [ColorMode.XYZ, ColorMode.RGB],
        (ColorMode.LAB, ColorMode.XYZ): [],
        (ColorMode.LAB, ColorMode.LUV): [ColorMode.XYZ],
        (ColorMode.LAB, ColorMode.LAB): [],
        (ColorMode.LAB, ColorMode.LCH_uv): [ColorMode.XYZ, ColorMode.LUV],
        
    }

    @staticmethod
    def rgb_to_hsv(r, g, b):
        return colorsys.rgb_to_hsv(r, g, b)

    @staticmethod
    def hsv_to_rgb(r, g, b):
        return colorsys.hsv_to_rgb(r, g, b)

    @staticmethod
    def hsl_to_hsv(h, s_hsl, l):
        v = l + s_hsl * min(l, 1 - l)
        s_hsv = 0 if v == 0 else 2 - 2 * l / v
        return (h, s_hsv, v)

    @staticmethod
    def hsv_to_hsl(h, s_hsv, v):
        l = v - v * s_hsv / 2
        s_hsl = 0 if l in (0, 1) else (v - l) / min(l, 1 - l)
        return (h, s_hsl, l)

    @staticmethod
    def rgb_to_hsl(r, g, b):
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return (h, s, l)

    @staticmethod
    def hsl_to_rgb(h, s, l):
        return colorsys.hls_to_rgb(h, l, s)

    @staticmethod
    def rgb_to_xyz(r, g, b):
        def _gamma(v):
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
        rgb = (_gamma(r), _gamma(g), _gamma(b))
        matrix = [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
        return tuple(
            sum(rgb[i] * c for i, c in enumerate(row))
            for row in matrix
        )

    @staticmethod
    def xyz_to_rgb(x, y, z):
        def _gamma1(v):
            return _clamp(v * 12.92 if v <= 0.0031308 else v ** (1/2.4) * 1.055 - 0.055)
        matrix = [
            [+3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, +1.8760108, +0.0415560],
            [+0.0556434, -0.2040259, +1.0572252],
        ]
        xyz = (x, y, z)
        return tuple(map(_gamma1, (
            sum(xyz[i] * c for i, c in enumerate(row))
            for row in matrix
        )))

    @staticmethod
    def xyz_to_luv(x, y, z):
        u1r = 0.2009
        v1r = 0.4610
        yr = 100

        kap = (29/3)**3
        eps = (6/29)**3

        try:
            u1 = 4*x / (x + 15*y + 3*z)
            v1 = 9*y / (x + 15*y + 3*z)
        except ZeroDivisionError:
            return 0, 0, 0

        y_r = y/yr
        l = 166 * y_r ** (1/3) - 16 if y_r > eps else kap * y_r
        u = 13 * l * (u1 - u1r)
        v = 13 * l * (v1 - v1r)
        return l, u, v

    @staticmethod
    def luv_to_xyz(l, u, v):
        u1r = 0.2009
        v1r = 0.4610
        yr = 100

        kap = (29/3)**3

        if l == 0:
            u1 = u1r
            v1 = v1r
        else:
            u1 = u / (13 * l) + u1r
            v1 = v / (13 * l) + v1r

        y = yr * l / kap if l <= 8 else yr * ((l + 16) / 116) ** 3
        x = y * 9*u1 / (4*v1)
        z = y * (12 - 3*u1 - 20*v1) / (4*v1)
        return x, y, z

    @staticmethod
    def luv_to_lch_uv(l, u, v):
        c = math.hypot(u, v)
        h = math.atan2(v, u)
        if h < 0:
            h += math.tau
        return l, c, h

    @staticmethod
    def lch_uv_to_luv(l, c, h):
        u = math.cos(h) * c
        v = math.sin(h) * c
        return l, u, v

    @staticmethod
    def xyz_to_lab(x, y, z):
        # D65 Illuminant aka sRGB(1,1,1)
        xn = 0.950489
        yn = 1
        zn = 108.8840

        delta = 6 / 29

        def f(t):
            return t ** (1/3) if t > delta ** 3 else t / (3*delta**2) + 4/29

        fy = f(y/yn)
        l = 116 * fy - 16
        a = 500 * (f(x/xn) - fy)
        b = 200 * (fy - f(z/zn))

        return l, a, b

    @staticmethod
    def lab_to_xyz(l, a, b):
        # D65 Illuminant aka sRGB(1,1,1)
        xn = 0.950489
        yn = 1
        zn = 108.8840

        delta = 6 / 29

        def f1(t):
            return t**3 if t > delta else 3*delta**2*(t-4/29)

        l1 = (l+16) / 116
        x = xn * f1(l1+a/500)
        y = yn * f1(l1)
        z = zn * f1(l1-b/200)

        return x, y, z


    @staticmethod
    def conv_func(mode_from, mode_to):
        return getattr(ParseColor, "%s_to_%s" % (mode_from.name.lower(), mode_to.name.lower()), None)

    @staticmethod
    def convert(tuple, mode_from, mode_to):
        if mode_from == mode_to:
            return tuple

        func = ParseColor.conv_func(mode_from, mode_to)
        if func:
            return func(*tuple)

        if (mode_from, mode_to) in ParseColor._conv_paths:
            steps = ParseColor._conv_paths[(mode_from, mode_to)] + [mode_to]
            for step in steps:
                func = ParseColor.conv_func(mode_from, step)
                if not func:
                    raise ValueError("Missing definition for conversion from %s to %s" % (mode_from, step))
                tuple = func(*tuple)
                mode_from = step
            return tuple

        raise ValueError("No conversion path from %s to %s" % (mode_from, mode_to))


class ManagedColor:
    Mode = ColorMode

    def __init__(self, c1, c2, c3, mode=ColorMode.RGB):
        self.vector = Vector(c1, c2, c3)
        self._mode = mode

    @property
    def mode(self):
        return self._mode

    def convert(self, v):
        if v == self._mode:
            return self

        self.vector = Vector(*ParseColor.convert(self.vector, self._mode, v))

        self._mode = v
        return self

    def clone(self):
        return ManagedColor(*self.vector, self._mode)

    def converted(self, mode):
        return self.clone().convert(mode)

    def to_color(self):
        return self.converted(ColorMode.RGB).vector

    @classmethod
    def from_color(cls, color):
        return cls(*color[:3], ColorMode.RGB)

    def __repr__(self):
        return "<%s %s [%.3f, %.3f, %.3f]>" % (
            (self.__class__.__name__, self.mode.name) + tuple(self.vector.components)
        )

    def _attrindex(self, name):
        comps = None

        if self._mode == ColorMode.RGB:
            comps = ({"r", "red"}, {"g", "green"}, {"b", "blue"})
        elif self._mode == ColorMode.HSV:
            comps = ({"h", "hue"}, {"s", "saturation"}, {"v", "value"})
        elif self._mode == ColorMode.HSL:
            comps = ({"h", "hue"}, {"s", "saturation"}, {"l", "lightness"})
        elif self._mode == ColorMode.LCH_uv: #in (ColorMode.LCH_uv, ColorMode.LCH_ab):
            comps = ({"l", "luma", "luminance"}, {"c", "choma"}, {"h", "hue"})
        elif self._mode == ColorMode.XYZ:
            comps = "xyz"
        elif self._mode == ColorMode.LUV:
            comps = "luv"
        elif self._mode == ColorMode.LAB:
            comps = "lab"

        if comps:
            for i, vals in enumerate(comps):
                if name in vals:
                    return i

        return None

    def __getattr__(self, name):
        if name not in vars(self) and name not in {"_mode", "vector"}:
            i = self._attrindex(name)
            if i is not None:
                return self.vector[i]
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name not in vars(self) and name not in {"_mode", "vector"}:
            i = self._attrindex(name)
            if i is not None:
                self.vector[i] = value
                return
        return super().__setattr__(name, value)


class ProcessColor:
    """Represents a color."""
    def __init__(self, *args, color_mode=None, normed=False, **kwargs):
        if color_mode is None:
            color_mode = color_parse_mode

        if (len(args) == 1) and isinstance(args[0], Color):
            r = args[0]._red
            g = args[0]._green
            b = args[0]._blue
            a = args[0]._alpha
        elif len(args) == 2 and isinstance(args[0], Color):
            r = args[0]._red
            g = args[0]._green
            b = args[0]._blue
            a = args[1]
        else:
            r, g, b, a = parse_color(*args, color_mode=color_mode,
                                     normed=normed, **kwargs)

        self._red = r
        self._green = g
        self._blue = b
        self._alpha = a

        self._recompute_hsb()

    def _recompute_rgb(self):
        """Recompute the RGB values from HSB values."""
        r, g, b = colorsys.hsv_to_rgb(self._hue, self._saturation, self._brightness)
        self._red = r
        self._greeen = g
        self._blue = b

    def _recompute_hsb(self):
        """Recompute the HSB values from the RGB values."""
        h, s, b = colorsys.rgb_to_hsv(self._red, self._green, self._blue)
        self._hue = h
        self._saturation = s
        self._brightness = b

    def lerp(self, target, amount):
        """Linearly interpolate one color to another by the given amount.

        :param target: The target color to lerp to.
        :type target: Color

        :param amount: The amount by which the color should be lerped
            (should be a float between 0 and 1).
        :type amount: float

        :returns: A new color lerped between the current color and the
            other color.
        :rtype: Color

        """
        lerped = (lerp(s, t, amount) for s, t in zip(self.rgba, target.rgba))
        return Color(*lerped, color_mode='RGB')

    def __repr__(self):
        fvalues = self._red, self._green, self._blue
        return "Color( red={}, green={}, blue={} )".format(*fvalues)

    __str__ = __repr__

    def __eq__(self, other):
        return all(math.isclose(sc, oc)
                   for sc, oc in zip(self.normalized, other.normalized))

    def __neq__(self, other):
        return not all(math.isclose(sc, oc)
                       for sc, oc in zip(self.normalized, other.normalized))

    @property
    def normalized(self):
        """Normalized RGBA color values"""
        return (self._red, self._green, self._blue, self._alpha)

    @property
    def gray(self):
        """The gray-scale value of the color.

        Performs a luminance conversion of the current color to
        grayscale.

        """
        # The formula we use to convert to grayscale is approximate
        # and probably not as accurate as a proper coloremetric
        # conversion. However, the number of calculations required is
        # less and GIMP uses something similar so we should be fine.
        #
        # REFERENCES:
        #
        # - "Converting Color Images to B&W"
        #   <https://www.gimp.org/tutorials/Color2BW/>
        #
        # - "Luma Coding in Video Systems" from "Grayscale"
        #   <https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems>
        #
        # - Wikipedia : Grayscale
        #   <https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale>
        #
        # - Conversion to grayscale, sample implementation (StackOverflow)
        # <https://stackoverflow.com/a/15686412>
        norm_gray  = 0.299 * self._red + 0.587 * self._green + 0.144 * self._blue
        return norm_gray * 255

    @gray.setter
    def gray(self, value):
        value = constrain(value / 255, 0, 1)
        self._red = value
        self._green = value
        self._blue = value
        self._recompute_hsb()

    @property
    def alpha(self):
        """The alpha value for the color."""
        return self._alpha * color_range[3]

    @alpha.setter
    def alpha(self, value):
        self._alpha = constrain(value / color_range[3], 0, 1)

    @property
    def rgb(self):
        """
        :returns: Color components in RGB.
        :rtype: tuple
        """
        return (self.red, self.green, self.blue)

    @property
    def rgba(self):
        """
        :returns: Color components in RGBA.
        :rtype: tuple
        """
        return (self.red, self.green, self.blue, self.alpha)


    @property
    def red(self):
        """The red component of the color"""
        return self._red * color_range[0]

    @red.setter
    def red(self, value):
        self._red = constrain(value / color_range[0], 0, 1)
        self._recompute_hsb()

    @property
    def green(self):
        """The green component of the color"""
        return self._green * color_range[1]

    @green.setter
    def green(self, value):
        self._green = constrain(value / color_range[1], 0, 1)
        self._recompute_hsb()

    @property
    def blue(self):
        """The blue component of the color"""
        return self._blue * color_range[2]

    @blue.setter
    def blue(self, value):
        self._blue = constrain(value / color_range[2], 0, 1)
        self._recompute_hsb()

    @property
    def hsb(self):
        """
        :returns: Color components in HSB.
        :rtype: tuple
        """
        return (self._hue, self._saturation, self._brightness)

    @property
    def hsba(self):
        """
        :returns: Color components in HSBA.
        :rtype: tuple
        """
        return (self.hue, self.saturation, self.brightness, self.alpha)

    @property
    def hue(self):
        """The hue component of the color"""
        return self._hue * color_range[0]

    @hue.setter
    def hue(self, value):
        self._hue = constrain(value / color_range[0], 0, 1)
        self._recompute_rgb()

    @property
    def saturation(self):
        """The saturation component of the color"""
        return self._saturation * color_range[1]

    @saturation.setter
    def saturation(self, value):
        self._saturation = constrain(value / color_range[1], 0, 1)
        self._recompute_rgb()

    @property
    def brightness(self):
        """The brightness component of the color"""
        return self._brightness * color_range[2]

    @brightness.setter
    def brightness(self, value):
        self._brightness = constrain(value / color_range[2], 0, 1)
        self._recompute_rgb()

    # ...and some convenient aliases
    r = red
    g = green
    h = hue
    s = saturation
    value = brightness
    v = value

    # `b` is tricky. depending on the current color code, this could
    # either be the brightness value or the blue value.
    @property
    def b(self):
        """The blue or the brightness value (depending on the color mode)."""
        if color_parse_mode== 'RGB':
            return self.blue
        elif color_parse_mode== 'HSB':
            return self.brightness
        else:
            raise ValueError("Unknown color mode {}".format(color_parse_mode))

    @b.setter
    def b(self, value):
        if color_parse_mode== 'RGB':
            self.blue = value
        elif color_parse_mode== 'HSB':
            self.brightness = value
        else:
            raise ValueError("Unknown color mode {}".format(color_parse_mode))

    @property
    def hex(self):
        """
        :returns: Color as a hex value
        :rtype: str
        """
        raise NotImple