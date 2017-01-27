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
from random import random
import IPython

#CONSTANTS:
N = 10
X = pow(2,N)
Y = pow(2,N)
START = np.array([0.5,0.5])
START_SQ = START ** 2
OUTPUT_DIR = "output"
OUTPUT_FILE = "hyphae"
bounds = [0,0,1,1]
DRAW_STEP = 10
HYPHAE_CIRC = 0.3
SPLIT_CHANCE = 0.2
SPLIT_ANGLE = math.pi * 0.4
NODE_START_SIZE = 0.005
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
frontier = deque()
#Root node:
root = None

#CAIRO: --------------------
logging.info("Setting up Cairo")
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, X,Y)
ctx = cairo.Context(surface)
ctx.scale(X,Y) #coords in 0-1 range

#Utility functions:
def createNode(location,d):
    logging.info("Creating a node for location: {}".format(location))
    global allNodes, graph, frontier, qtree
    newNode = {'loc':location, 'd':d,'uuid': uuid1()}
    allNodes[newNode['uuid']] = newNode
    graph.add_node(newNode['uuid'])
    frontier.append(newNode['uuid'])
    qtree.insert(item=newNode['uuid'],bbox=bboxFromNode(newNode))
    return newNode

def bboxFromNode(node):
    bbox = [
        node['loc'][0]-node['d'],
        node['loc'][1]-node['d'],
        node['loc'][0]+node['d'],
        node['loc'][1]+node['d']
    ]
    return bbox

def initialise():
    logging.info("Setting up root node")
    global root
    root = createNode(START,NODE_START_SIZE)


def getNeighbourhood(x,y):
    """
    for a given new node, get nodes local to it spatially
    + the predecessor chain
    """
    delta = NODE_START_SIZE * 0.5
    bbox = [x-delta,y-delta,
            x+delta,y+delta]
    logging.info("Neighbourhood of ({},{}) -> bbox {}".format(x,y, bbox))
    matches = qtree.intersect(bbox)
    return matches

    

def getPredecessorUUIDS(nodeUUID):
    """
    Get the tree predecessor of a node
    """
    return graph.predecessors(nodeUUID)
    

def doNodePairsIntersect(node1a,node1b,node2a,node2b):
    """
    See if two pairs of nodes cross each other
    """
    return False


def allFrontiersAreAtBoundary():
    logging.info("Checking {} frontiers are at boundary".format(len(frontier)))
    #distance check all nodes in the frontier
    #  d(START_NODE,FRONTIER_NODE) < HYPHAE_CIRC
    nodesFromUUIDS = [allNodes[x] for x in frontier]
    distances = [utils.get_distance_raw(x['loc'],START) for x in nodesFromUUIDS]
    logging.info("Distances: {}".format(distances))
    distanceSufficientFromCentre = [x > HYPHAE_CIRC for x in distances]
    logging.info("Sufficient: {}".format(distanceSufficientFromCentre))
    value = all(distanceSufficientFromCentre)
    logging.info("It is {} that all frontier nodes are at boundary".format(value))
    return value



#Main Growth function:
def grow():
    global graph
    logging.info("Growing")
    #pick a frontier node
    focusNodeUUID = frontier.popleft()
    focusNode = allNodes[focusNodeUUID]
    #get its predecessor
    predecessorUUIDS = getPredecessorUUIDS(focusNodeUUID)
    if len(predecessorUUIDS) == 0:
        #no predecessor, pick a random direction
        logging.info("No predecessor, picking random direction")
        #todo: rotate around the point
        rndVec = focusNode['loc'] + np.random.random(2) - 0.5
        normalized = utils.get_normal(focusNode['loc'],rndVec)
        newPoint = focusNode['loc'] + (normalized * (NODE_START_SIZE * 1.2))
    else:
        logging.info("Extending vector")
        #create a vector out of the pair / Alt: move -> d(p,x) < d(n,x)
        predecessor = allNodes[predecessorUUIDS[0]]
        normalized = utils.get_normal(predecessor['loc'],focusNode['loc'])
        newPoint = focusNode['loc'] + (normalized * (NODE_START_SIZE * 1.2))
        #todo: add wiggle
        
    #move along that vector
    #check for intersections and being too close
    intersections = [x for x in getNeighbourhood(*newPoint) if x not in predecessorUUIDS and x != focusNodeUUID]
    if len(intersections) != 0:
        logging.info("There are {} intersections, not adding a new node".format(len(intersections)))
        return True
        #while fail the above check, wiggle the vector
    
    #Split branch based on split chance
    if random() > SPLIT_CHANCE:
        newPositions = [newPoint]
    else:
        newPositions = [newPoint]
    #add new node/nodes to frontier,
    #node pairs are mirrored either side of the canonical next node

    #create the nodes
    newNodes = [createNode(x,NODE_START_SIZE) for x in newPositions]
    for x in newNodes:
        graph.add_edge(focusNodeUUID,x['uuid'])
            
    
    return False

def draw_hyphae():
    logging.info("Drawing")
    #clear the context
    utils.clear_canvas(ctx)
    #from the root node of the graph
    nodes = deque([root['uuid']])
    #BFS the tree
    i = 1
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        #todo: get a line between currentNode and predecessor
        #draw the node / line
        ctx.set_source_rgba(*[1,i,1,1])
        logging.info("Circle: {:.2f}, {:.2f}".format(*currentNode['loc']))
        utils.drawCircle(ctx,*currentNode['loc'],currentNode['d'])
        #get it's children
        nodes.extend(graph.successors(currentUUID))
        if i == 0:
            i = 1
        else:
            i = 0
            
    return True


if __name__ == "__main__":
    logging.info('Starting main')
    initialise()
    i = 0
    growSaysFinish = False
    while not allFrontiersAreAtBoundary() and not growSaysFinish:
        growSaysFinish = grow()
        i += 1
        if i % DRAW_STEP == 0:
            draw_hyphae()
            utils.write_to_png(surface,join(OUTPUT_DIR,OUTPUT_FILE),i)
    draw_hyphae()
    utils.write_to_png(surface,join(OUTPUT_DIR,OUTPUT_FILE),"FINAL")
