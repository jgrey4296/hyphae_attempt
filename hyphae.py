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
from random import choice
from numpy.random import random
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
DRAW_STEP = 100
HYPHAE_CIRC = 0.4
WIGGLE_CHANCE = 0.4
WIGGLE_AMNT = 0.4
WIGGLE_VARIANCE = 0.4
SPLIT_CHANCE = 0.4
SPLIT_ANGLE = math.pi * 0.3
SPLIT_ANGLE_VARIANCE = 0.5
NODE_START_SIZE = 0.005
NODE_SIZE_DECAY = 0.00002
MIN_NODE_SIZE = 0.0009
SIZE_DIFF = 0.003
MAX_ATTEMPTS_PER_NODE = 4
LINE_PROPORTION_DISTORTION = 0.8
LINE_DISTORTION_UPSCALING = 100
BRANCH_BACKTRACK_AMNT = 0.3
MAX_FRONTIER_NODES = 100
NODE_OPACITY = 0.3
NODE_COLOUR = [1,1,1,NODE_OPACITY]
NEIGHBOUR_DELTA = 1.5
DRAW_NODES = False
ANIMATE = False

node_colour = lambda : np.concatenate((random(3),[NODE_OPACITY]))

MAIN_COLOUR = node_colour()

logging.info("Setting up Graph and QuadTree")
allNodes = {}
branchPoints = []
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
def createNode(location,d,distance=0):
    logging.debug("Creating a node for location: {}".format(location))
    global allNodes, graph, frontier, qtree
    newNode = {'loc':location, 'd':d,'uuid': uuid1(), 'remaining':MAX_ATTEMPTS_PER_NODE,
               'distance_from_branch': distance, 'perpendicular' : False}
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


def getNeighbourhood(x,y,d):
    """
    for a given new node, get nodes local to it spatially
    + the predecessor chain
    """
    delta = d*NEIGHBOUR_DELTA
    bbox = [x-delta,y-delta,
            x+delta,y+delta]
    logging.debug("Neighbourhood of ({},{}) -> bbox {}".format(x,y, bbox))
    matches = qtree.intersect(bbox)
    matchNodes = [allNodes[x] for x in matches]
    return matchNodes

    

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
    global frontier
    logging.debug("Checking {} frontiers are at boundary".format(len(frontier)))
    #distance check all nodes in the frontier
    nodesFromUUIDS = [allNodes[x] for x in frontier]
    distances = [utils.get_distance(x['loc'],START) for x in nodesFromUUIDS]
    paired = zip(frontier,distances)
    logging.debug("Distances: {}".format(distances))
    #filter the frontier:
    frontier = deque([x for x,y in paired if y < HYPHAE_CIRC])
    return len(frontier) == 0



#Main Growth function:
def grow():
    global graph
    logging.debug("Growing")
    #pick a frontier node
    focusNodeUUID = frontier.popleft()
    focusNode = allNodes[focusNodeUUID]
    #get its predecessor
    predecessorUUIDS = getPredecessorUUIDS(focusNodeUUID)
    if focusNode['perpendicular'] == True and len(predecessorUUIDS) > 0:
        #get the noram, rotate 90 degrees
        predecessor = allNodes[predecessorUUIDS[0]]
        normalized = utils.get_normal(predecessor['loc'],focusNode['loc'])
        direction = choice([ [[0,-1],[1,0]], [[0,1], [-1, 0]] ])
        perpendicular = normalized.dot(direction)
        newPoint = focusNode['loc'] + (perpendicular * (2*focusNode['d']))
    elif len(predecessorUUIDS) == 0:
        #no predecessor, pick a random direction
        logging.debug("No predecessor, picking random direction")
        #todo: rotate around the point
        rndVec = focusNode['loc'] + (np.random.random(2) - 0.5)
        normalized = utils.get_normal(focusNode['loc'],rndVec)
        newPoint = focusNode['loc'] + (normalized * (2 * focusNode['d']))
    else:
        logging.debug("Extending vector")
        #create a vector out of the pair / Alt: move -> d(p,x) < d(n,x)
        predecessor = allNodes[predecessorUUIDS[0]]
        normalized = utils.get_normal(predecessor['loc'],focusNode['loc'])
        newPoint = focusNode['loc'] + (normalized * (2*focusNode['d']))
        #todo: add wiggle
        if random() < WIGGLE_CHANCE:
            newPoint = utils.rotatePoint(focusNode['loc'],newPoint,
                                         radMin=-(WIGGLE_AMNT + WIGGLE_VARIANCE),
                                         radMax=(WIGGLE_AMNT + WIGGLE_VARIANCE))

    #Split branch based on split chance
    if random() < SPLIT_CHANCE:
        s1 = utils.rotatePoint(focusNode['loc'], newPoint,
                               radMin=-(SPLIT_ANGLE+SPLIT_ANGLE_VARIANCE),
                               radMax=-(SPLIT_ANGLE+SPLIT_ANGLE_VARIANCE))
        s2 = utils.rotatePoint(focusNode['loc'], newPoint,
                               radMin=SPLIT_ANGLE+SPLIT_ANGLE_VARIANCE,
                               radMax=SPLIT_ANGLE+SPLIT_ANGLE_VARIANCE)
        newPositions = [s1, s2]
        decay = NODE_SIZE_DECAY
        distance_from_branch = 0
        
        
    else:
        newPositions = [newPoint]
        decay = 0.0
        distance_from_branch = focusNode['distance_from_branch'] + 1

    #check for intersections and being too close
    neighbours = [x for newPos in newPositions for x in getNeighbourhood(*newPos,focusNode['d']) if x['uuid'] not in predecessorUUIDS and x['uuid'] != focusNodeUUID]
    #distances = [(utils.get_distance_raw(x['loc'],newPoint),x['d']) for x in neighbours]
    #too_close = [x for x,y in distances if x < ((y+y)**2)]
    if len(neighbours) != 0:
        logging.debug("There are {} intersections, not adding a new node".format(len(neighbours)))
        focusNode['remaining'] = focusNode['remaining']-1
        allNodes[focusNode['uuid']] = focusNode
        if focusNode['remaining'] > 0 and len(frontier) < MAX_FRONTIER_NODES:
            frontier.append(focusNode['uuid'])
        return False
    
    #add new node/nodes to frontier,
    #create the nodes
    newNodes = [createNode(x,focusNode['d']-decay,distance_from_branch) for x in newPositions]
    for x in newNodes:
        graph.add_edge(focusNodeUUID,x['uuid'])

    if len(newNodes) > 1:
        branchPoints.append(focusNode['uuid'])

    #occasionally backtrack from a branch point:
    if random() < BRANCH_BACKTRACK_AMNT and len(branchPoints) > 0:
        rndBranch = choice(branchPoints)
        rndBranchNode = allNodes[rndBranch]
        length_of_branch = rndBranchNode['distance_from_branch']
        midPoint = int(length_of_branch * 0.5)
        currentNodeUUID = rndBranch
        for x in range(midPoint):
            currentNodeUUID = graph.predecessors(currentNodeUUID)[0]

        potentialNode = allNodes[currentNodeUUID]
        if potentialNode['remaining'] > 0 and len(frontier) < MAX_FRONTIER_NODES :
            potentialNode['perpendicular'] = True
            frontier.append(currentNodeUUID)
            
    
    return False

def draw_hyphae():
    logging.debug("Drawing")
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
        logging.debug("Circle: {:.2f}, {:.2f}".format(*currentNode['loc']))
        utils.drawCircle(ctx,*currentNode['loc'],currentNode['d']-SIZE_DIFF)
        #get it's children
        nodes.extend(graph.successors(currentUUID))
        if i == 0:
            i = 1
        else:
            i = 0
            
    return True

def draw_hyphae_2():
    logging.debug("Drawing alternate")
    utils.clear_canvas(ctx)
    nodes = deque(graph.successors(root['uuid']))
    #BFS the tree:
    ctx.set_source_rgba(*MAIN_COLOUR)
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        prev = allNodes[graph.predecessors(currentNode['uuid'])[0]]
        points = utils.createLine(*currentNode['loc'], *prev['loc'],LINE_DISTORTION_UPSCALING)
        length_of_line = np.linalg.norm(points[-1] - points[0])
        distorted = utils.displace_along_line(points,length_of_line * LINE_PROPORTION_DISTORTION,LINE_DISTORTION_UPSCALING)
        nodes.extend(graph.successors(currentUUID))
        for x,y in distorted:
            utils.drawCircle(ctx,x,y,MIN_NODE_SIZE)
        #for x,y in points:
        #    utils.drawCircle(ctx,x,y,utils.clamp(currentNode['d']-SIZE_DIFF,MIN_NODE_SIZE,NODE_START_SIZE))

    return True

if __name__ == "__main__":
    logging.info('Starting main')
    initialise()
    i = 0
    growSaysFinish = False
    while not allFrontiersAreAtBoundary() and not growSaysFinish:
        i += 1
        growSaysFinish = grow()
        logging.info(i)
        if not ANIMATE:
            continue

        if i % DRAW_STEP == 0:
            if DRAW_NODES:
                draw_hyphae()
            else:
                draw_hyphae_2()
            utils.write_to_png(surface,join(OUTPUT_DIR,OUTPUT_FILE),i)
        if i % 50 == 0:
            logging.info("...")
    draw_hyphae()
    utils.write_to_png(surface,join(OUTPUT_DIR,OUTPUT_FILE),"FINAL")
