import random
import math
from .vector import NVector
from model.shapes import Path
from model import *
from model import easing
from model import properties


def shake(position_prop, x_radius, y_radius, start_time, end_time, n_frames, interp=easing.Linear()):
    if not isinstance(position_prop, list):
        position_prop = [position_prop]

    n_frames = int(round(n_frames))
    frame_time = (end_time - start_time) / n_frames
    startpoints = list(map(
        lambda pp: pp.get_value(start_time),
        position_prop
    ))

    for i in range(n_frames):
        x = (random.random() * 2 - 1) * x_radius
        y = (random.random() * 2 - 1) * y_radius
        for pp, start in zip(position_prop, startpoints):
            px = start[0] + x
            py = start[1] + y
            pp.add_keyframe(start_time + i * frame_time, Vector(px, py), interp)

    for pp, start in zip(position_prop, startpoints):
        pp.add_keyframe(end_time, start, interp)


def rot_shake(rotation_prop, angles, start_time, end_time, n_frames):
    frame_time = (end_time - start_time) / n_frames
    start = rotation_prop.get_value(start_time)

    first = 0
    for i in range(0, n_frames):
        a = angles[i % len(angles)] * math.sin(i/n_frames * math.pi)
        rotation_prop.add_keyframe(start_time + i * frame_time, start + a)
    rotation_prop.add_keyframe(end_time, start)


