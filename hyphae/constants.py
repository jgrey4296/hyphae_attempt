import numpy as np
from numpy.random import random
import math

##############################
#CONSTANTS:
####################
#Size of generated image:
N = 10
X = pow(2, N)
Y = pow(2, N)
#Start points points
CENTRE = np.array([0.5, 0.5])
START_NODES = np.array([[0.4, 0.4], [0.6, 0.6]])
#START_NODES = np.array([[0.5, 0.5]])
#Where to output the renderings:
OUTPUT_DIR = "output"
OUTPUT_FILE = "hyphae"
#Pickle output details:
PICKLE_DIR = "output_pickle"
PICKLE_FILENAME = "hyphae_graph"
#Render options:
DRAW_NODES = False
DRAW_LINES = True
DRAW_PATHS = False
ANIMATE = True
SIZE_DIFF = 0.003
LINE_PROPORTION_DISTORTION = 0.8
LINE_DISTORTION_UPSCALING = 100
NODE_OPACITY = 0.3
#NODE_COLOUR = [1, 1, 1, NODE_OPACITY]
NODE_COLOUR = lambda: np.concatenate((random(3), [NODE_OPACITY]))
MAIN_COLOUR = NODE_COLOUR()
DRAW_STEP = 1
LINE_WIDTH = 0.005
#BOUNDS of the quadtree:
BOUNDS = [0, 0, 1, 1]
#Neighbourhood distance size:
HYPHAE_CIRC = 0.5
NEIGHBOUR_DELTA = 1.4
#Growth rules:
WIGGLE_CHANCE = 0.3
WIGGLE_AMNT = 0.4
WIGGLE_VARIANCE = 0.8
SPLIT_CHANCE = 0.5
SPLIT_ANGLE = math.pi * 0.4
SPLIT_ANGLE_VARIANCE = 0.0
NODE_START_SIZE = 0.007
NODE_SIZE_DECAY = 0 #0.00002
MIN_NODE_SIZE = 0.0008
MAX_ATTEMPTS_PER_NODE = 4
BRANCH_BACKTRACK_AMNT = 0.8
BACKTRACK_STEP_NUM = lambda x: int(1 + (x-1 * random()))
MAX_FRONTIER_NODES = 100
MAX_GROWTH_STEPS = 10000
