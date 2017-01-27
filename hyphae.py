"""
An re-implementation of inconvergent's hyphae
"""
# Setup root_logger:
import logging as root_logger
LOGLEVEL = root_logger.DEBUG
logFileName = "hyphae.log"
root_logger.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')

console = root_logger.StreamHandler()
console.setLevel(root_logger.INFO)
root_logger.getLogger('').addHandler(console)
logging = root_logger.getLogger(__name__)

from os.path import join
import numpy as np
import pandas
import networkx as nx
import cairocffi as cairo
import utils
from collections import namedtuple, deque
import pyqtree
import math
from uuid import uuid1

#CONSTANTS:
N = 8
X = pow(2,N)
Y = pow(2,N)
START = (0.5,0.5)
OUTPUT_DIR = "output"
OUTPUT_FILE = "hyphae"
bounds = [0,0,1,1]
DRAW_STEP = 10
HYPHAE_CIRC = 0.4
SPLIT_CHANCE = 0.2
SPLIT_ANGLE = math.pi * 0.4
NODE_START_SIZE = 0.05
VERIFY_LAMBDA = 0.1

def verify_bbox(x,y):
    return [x-VERIFY_LAMBDA,
            y-VERIFY_LAMBDA,
            x+VERIFY_LAMBDA,
            y+VERIFY_LAMBDA]


logging.info("Setting up Graph and QuadTree")
allNodes = {}
#Keep track of nodes in a directed graph
graph = nx.DiGraph()
#check for neighbours with the qtree 
qtree = pyqtree.Index(bbox=bounds)
#qtree.insert(item=item, bbox=item.bbox)
#matches = qtree.intersect(bbox)
frontier = []
#Root node:
root = None

#CAIRO: --------------------
logging.info("Setting up Cairo")
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, X,Y)
ctx = cairo.Context(surface)
ctx.scale(X,Y) #coords in 0-1 range

#Utility functions:

def bboxFromNode(node):
    bbox = [
        node['x']-node['d'],
        node['y']-node['d'],
        node['x']+node['d'],
        node['y']+node['d']
    ]
    return bbox

def initialise():
    global root
    root = {'x': START[0], 'y':START[1], 'd':NODE_START_SIZE, 'uuid' : uuid1() }
    #Store the node in all nodes:
    allNodes[root['uuid']] = root
    #Add the root node to the graph
    graph.add_node(root['uuid'])
    #add it to the frontier
    frontier.append(root['uuid'])
    #add it to the quad tree
    qtree.insert(item=root['uuid'],bbox=bboxFromNode(root))
    return True


def getNeighbourhood(node_coords):
    """
    for a given new node, get nodes local to it spatially
    + the predecessor chain
    """
    return []

def getPredecessor(node):
    """
    Get the tree predecessor of a node
    """
    return None

def doNodePairsIntersect(node1a,node1b,node2a,node2b):
    """
    See if two pairs of nodes cross each other
    """
    return False


def allFrontiersAreAtBoundary():
    #distance check all nodes in the frontier
    #  d(START_NODE,FRONTIER_NODE) < HYPHAE_CIRC
    
    return True

#Main Growth function:
def grow():
    #pick a frontier node
    #get its predecessor
    #create a vector out of the pair / Alt: move -> d(p,x) < d(n,x) 
    #move along that vector
    #check for intersections and being too close

    #while fail the above check, wiggle the vector
    
    #Split branch based on split chance

    #add new node/nodes to frontier,
    #node pairs are mirrored either side of the canonical next node

    #add nodes into the graph, and the quadtree
    #store the node size in the graph
    
    
    return False

def draw_hyphae():
    #clear the context
    utils.clear_canvas(ctx)
    #from the root node of the graph
    nodes = deque([root['uuid']])
    #BFS the tree
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        #draw the node
        ctx.set_source_rgba(*[0.2,0.8,0.1,1])
        utils.drawCircle(ctx,currentNode['x'],currentNode['y'],currentNode['d'])
        #get it's children
        nodes.extend(graph.successors(currentUUID))
    
    return True


if __name__ == "__main__":
    logging.info('Starting main')
    initialise()
    i = 0
    while not allFrontiersAreAtBoundary():
        grow()
        i += 1
        if i % DRAW_STEP:
            draw_hyphae()
            utils.write_to_png(surface,join(OUTPUT_DIR,OUTPUT_FILE),i)
    draw_hyphae()
    utils.write_to_png(surface,join(OUTPUT_DIR,OUTPUT_FILE),"FINAL")
