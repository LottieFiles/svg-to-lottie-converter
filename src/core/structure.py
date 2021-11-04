from contextlib import contextmanager
from utils import color
from core import base

@contextmanager
def push_style():
    prev_fill_color = base.renderer.fill_color
    prev_fill_enabled = base.renderer.fill_enabled
    prev_stroke_enabled = base.renderer.stroke_enabled
    prev_stroke_color = base.renderer.stroke_color
    prev_tint_color = base.renderer.tint_color
    prev_tint_enabled = base.renderer.tint_enabled

    prev_ellipse_mode = primitives._ellipse_mode
    prev_rect_mode = primitives._rect_mode
    prev_shape_mode = primitives._shape_mode

    prev_color_mode = color.color_parse_mode
    prev_color_range = color.color_range

    yield

    base.renderer.fill_color = prev_fill_color
    base.renderer.fill_enabled = prev_fill_enabled
    base.renderer.stroke_color = prev_stroke_color
    base.renderer.stroke_enabled = prev_stroke_enabled
    base.renderer.tint_color = prev_tint_color
    base.renderer.tint_enabled = prev_tint_enabled

    primitives._ellipse_mode = prev_ellipse_mode
    primitives._rect_mode = prev_rect_mode
    primitives._shape_mode = prev_shape_mode

    color.prev_color_parse_mode = prev_color_mode
    color.prev_color_range = prev_color_range

def pop_style():
    push_style()