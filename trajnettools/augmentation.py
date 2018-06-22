import math
import random

from .data import Row


def random_rotation(path):
    theta = random.random() * 2 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    return [Row(r.frame, r.pedestrian, ct * r.x + st * r.y, -st * r.x + ct * r.y)
            for r in path]
