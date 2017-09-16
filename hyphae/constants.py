import numpy as np
from random import randrange
from numpy.random import random
import numpy.random as rand
import math
from cairo_utils import clamp, get_distance

##############################
#CONSTANTS:
####################
#Size of generated image:
N = 10
#Start points points
CENTRE = np.array([0.5, 0.5])
#The Radius of the circle to hyphae in
HYPHAE_CIRC = 0.45
#Default initial nodes to grow from:
#START_NODES = np.array([[0.4, 0.4], [0.6, 0.6]])
NUM_START=1
START_NODES = []

while len(START_NODES) < NUM_START:
    loc = random(2)
    if get_distance(loc, CENTRE) < (HYPHAE_CIRC * 0.5):
        START_NODES.append(loc)

#Where to output the renderings:
OUTPUT_DIR = "imgs"
#Pickle output details:
PICKLE_DIR = "output_pickle"
PICKLE_FILENAME = "hyphae_graph"

#Render options:
SIZE_DIFF = 0.003
LINE_PROPORTION_DISTORTION = 0.8
LINE_DISTORTION_UPSCALING = 100
NODE_ALPHA = 0.3
NODE_COLOUR = lambda: np.concatenate((random(3), [NODE_ALPHA]))
MAIN_COLOUR = NODE_COLOUR()
LINE_WIDTH = 0.005

#BOUNDS of the quadtree:
BOUNDS = [0, 0, 1, 1]


#Interpolation and clamping
rad_lerp = lambda x: np.interp(x, [0, 1], [-math.pi, math.pi])
rad_clamp = lambda x: clamp(x, -math.pi, math.pi)

NODE_START_SIZE = 0.007
NODE_SIZE_DECAY = 0 #0.00002
MIN_NODE_SIZE = 0.0008


MAX_FRONTIER_NODES = 100
MAX_GROWTH_STEPS = 200000


randf = lambda x=1: rand.beta(1,5,x)

####################
#Main Values to modify:
BRANCH_BACKTRACK_AMNT = 0.1
backtrack_attempts = 20
MAX_ATTEMPTS_PER_NODE = 5
MIN_BRANCH_LENGTH = 10
BACKTRACK_LIKELIHOOD = 0.4
#Initial Growth rules:
WIGGLE_CHANCE = 0.5
WIGGLE_AMNT = math.pi * 0
WIGGLE_VARIANCE = math.pi * 0.4

SPLIT_CHANCE = 0.1
SPLIT_ANGLE = math.pi * 0.3
SPLIT_ANGLE_VARIANCE = math.pi * 0.7

# 0.001 - 0.02
NEIGHBOUR_DELTA = 0.0001
DELTA_CLAMP = (0.0001, 0.02)

#Mutation ranges:
#mod (colour, delta, wiggle, split, branch)
mod_chances = (0.4,0.2,0.5,0.5,0)

mut_c_range = (-0.0,0.1)
mut_d_range = (-0.001,0.001)

#chance, amnt, variance ranges
mut_wcr = (-0.0, 0.05)
mut_war = (0,0.2)
mut_wvr = (0,0.2)

mut_scr = (-0.001, 0.01)
mut_sar = (0,0.2)
mut_svr = (0,0.2)

mut_btc = (0.0, 0.02)
