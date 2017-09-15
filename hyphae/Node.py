from uuid import uuid1
from .constants import MAX_ATTEMPTS_PER_NODE, NEIGHBOUR_DELTA
from cairo_utils.math import get_distance, get_normal
import numpy as np

class Node:
    """ A Single Point in the Hyphae Calculation """

    def __init__(self, loc, d, distance, perpendicular=False, colour=0):
        assert(len(loc) == 2)
        self.id = uuid1()
        self.loc = loc
        self.d = d #diameter
        self.distance = distance
        self.remaining = MAX_ATTEMPTS_PER_NODE
        self.perpendicular = perpendicular
        self.distance_from_branch = 0
        self.colour = colour
        self.bbox = np.row_stack((self.loc - self.d,
                                  self.loc + self.d))

    def get_delta_bbox(self, delta=NEIGHBOUR_DELTA):
        delta *= self.d
        return np.row_stack((self.loc - delta,
                             self.loc + self.d))

    def distance_to(self, point):
        return get_distance(self.loc, point)

    def get_normal(self, other):
        assert(isinstance(other, Node))
        return get_normal(self.loc, other.loc)

    def get_random_vec(self):
        return get_normal(self.loc, self.loc + (np.random.random(2) - 0.5))

    def move(self, dir=None):
        if dir is None:
            dir = self.get_random_vec()
        return self.loc + (dir * (2 * self.d))

    def attempt(self):
        if self.open():
            self.remaining -= 1

    def open(self):
        return bool(self.remaining)
