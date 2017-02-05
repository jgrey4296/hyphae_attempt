"""
An re-implementation of inconvergent's hyphae
"""
# Setup root_logger:
import logging as root_logger
LOGLEVEL = root_logger.DEBUG
LOG_FILE_NAME = "hyphae.log"
root_logger.basicConfig(filename=LOG_FILE_NAME, level=LOGLEVEL, filemode='w')

CONSOLE = root_logger.StreamHandler()
CONSOLE.setLevel(root_logger.INFO)
root_logger.getLogger('').addHandler(CONSOLE)
logging = root_logger.getLogger(__name__)

from os.path import join
from collections import namedtuple, deque
import math
from random import choice
import numpy as np
from numpy.random import random
import networkx as nx
import cairo
import utils
import pyqtree
from uuid import uuid1
import IPython
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
###############################
#Variables:
####################
logging.info("Setting up Graph and QuadTree")
allNodes = {}
branchPoints = []
#Keep track of nodes in a directed graph
graph = nx.DiGraph()
#check for neighbours with the qtree
qtree = pyqtree.Index(bbox=BOUNDS)
#qtree.insert(item=item, bbox=item.bbox)
#matches = qtree.intersect(bbox)
frontier = deque()
#Root nodes:
root_nodes = []
#Colours :
colours = [NODE_COLOUR() for x in START_NODES]
##############################
#CAIRO:
####################
logging.info("Setting up Cairo")
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, X, Y)
ctx = cairo.Context(surface)
ctx.scale(X, Y) #coords in 0-1 range
##############################
#Utility functions:
####################
def create_node(location, d, distance=0, colour=0):
    """
    Create a node, add it to the graph, frontier, and quadtree
    """
    logging.debug("Creating a node for location: {}".format(location))
    if colour > len(colours):
        raise Exception("Tried to create a node with a non-existent colour")

    newNode = {'loc':location, 'd':d, 'uuid': uuid1(), 'remaining':MAX_ATTEMPTS_PER_NODE,
               'distance_from_branch': distance, 'perpendicular' : False, 'colour' : colour}
    allNodes[newNode['uuid']] = newNode
    graph.add_node(newNode['uuid'])
    frontier.append(newNode['uuid'])
    qtree.insert(item=newNode['uuid'], bbox=bbox_from_node(newNode))
    return newNode

def bbox_from_node(node):
    """ Given a node, return a bounding box based on its d / size """
    bbox = [
        node['loc'][0]-node['d'],
        node['loc'][1]-node['d'],
        node['loc'][0]+node['d'],
        node['loc'][1]+node['d']
    ]
    return bbox

def get_neighbourhood(x, y, d):
    """
    for a given new node, get nodes local to it spatially
    + the predecessor chain
    """
    delta = d*NEIGHBOUR_DELTA
    bbox = [x-delta, y-delta,
            x+delta, y+delta]
    logging.debug("Neighbourhood of ({},{}) -> bbox {}".format(x, y, bbox))
    matches = qtree.intersect(bbox)
    match_nodes = [allNodes[x] for x in matches]
    return match_nodes

def all_frontiers_are_at_boundary():
    """ verify the distances of the frontier to the centre """
    global frontier
    logging.debug("Checking {} frontiers are at boundary".format(len(frontier)))
    #distance check all nodes in the frontier
    nodesFromUUIDS = [allNodes[x] for x in frontier]
    distances = [utils.get_distance(x['loc'], CENTRE) for x in nodesFromUUIDS]
    paired = zip(frontier, distances)
    logging.debug("Distances: {}".format(distances))
    #filter the frontier:
    frontier = deque([x for x, y in paired if y < HYPHAE_CIRC])
    return len(frontier) == 0

def determine_new_point(node):
    """
    Given a node, figure out the node that follows location
    """
    predecessorUUIDs = graph.predecessors(node['uuid'])
    if node['perpendicular'] is True and len(predecessorUUIDs) > 0:
        #get the noram, rotate 90 degrees
        predecessor = allNodes[predecessorUUIDs[0]]
        normalized = utils.get_normal(predecessor['loc'], node['loc'])
        direction = choice([[[0, -1], [1, 0]], [[0, 1], [-1, 0]]])
        perpendicular = normalized.dot(direction)
        newPoint = node['loc'] + (perpendicular * (2*node['d']))
    elif len(predecessorUUIDs) == 0:
        #no predecessor, pick a random direction
        logging.debug("No predecessor, picking random direction")
        #todo: rotate around the point
        rndVec = node['loc'] + (np.random.random(2) - 0.5)
        normalized = utils.get_normal(node['loc'], rndVec)
        newPoint = node['loc'] + (normalized * (2 * node['d']))
    else:
        logging.debug("Extending vector")
        #create a vector out of the pair / Alt: move -> d(p, x) < d(n, x)
        predecessor = allNodes[predecessorUUIDs[0]]
        normalized = utils.get_normal(predecessor['loc'], node['loc'])
        newPoint = node['loc'] + (normalized * (2*node['d']))
        if random() < WIGGLE_CHANCE:
            newPoint = utils.rotatePoint(node['loc'], newPoint, 
                                         radMin=-(WIGGLE_AMNT + WIGGLE_VARIANCE), 
                                         radMax=(WIGGLE_AMNT + WIGGLE_VARIANCE))

    return newPoint

def split_if_necessary(point, focusNode):
    """ Split branch based on split chance """
    if not focusNode['perpendicular'] and random() < SPLIT_CHANCE:
        s1 = utils.rotatePoint(focusNode['loc'], point, 
                               radMin=-(SPLIT_ANGLE+SPLIT_ANGLE_VARIANCE), 
                               radMax=-(SPLIT_ANGLE-SPLIT_ANGLE_VARIANCE))
        s2 = utils.rotatePoint(focusNode['loc'], point, 
                               radMin=SPLIT_ANGLE-SPLIT_ANGLE_VARIANCE, 
                               radMax=SPLIT_ANGLE+SPLIT_ANGLE_VARIANCE)
        newPositions = [s1, s2]
        decay = NODE_SIZE_DECAY
        distance_from_branch = 0
    elif focusNode['perpendicular']:
        newPositions = [point]
        decay = 0.0
        distance_from_branch = 0
    else:
        newPositions = [point]
        decay = 0.0
        distance_from_branch = focusNode['distance_from_branch'] + 1

    return (newPositions, decay, distance_from_branch)


def positions_collide(positions, focusNode):
    """ 
    See if the positions specified are too close to any existing nodes
    """
    predecessorUUIDs = graph.predecessors(focusNode['uuid'])
    neighbours = [x for newPos in positions for x in get_neighbourhood(*newPos, focusNode['d']) \
                  if x['uuid'] not in predecessorUUIDs and x['uuid'] != focusNode['uuid']]
    if len(neighbours) != 0:
        logging.debug("There are {} intersections,  not adding a new node".format(len(neighbours)))
        focusNode['remaining'] = focusNode['remaining']-1
        allNodes[focusNode['uuid']] = focusNode
        if focusNode['remaining'] > 0 and len(frontier) < MAX_FRONTIER_NODES:
            frontier.append(focusNode['uuid'])
        return True
   
    return False    
  

def grow_suitable_nodes(newPositions, decay, distance_from_branch, focusNode):
    """
    Create new nodes,  storing any branch points,  and linking the edge to its parent
    """
    #retrieve or create a colour:
    if focusNode['perpendicular']:
        colours.append(NODE_COLOUR())
        colour_index = len(colours) - 1
    else:
        colour_index = focusNode['colour']
    #create the nodes:
    newNodes = [create_node(x, focusNode['d']-decay, distance_from_branch, colour_index) \
                for x in newPositions]
    #add the nodes to the graph:
    for x in newNodes:
        graph.add_edge(focusNode['uuid'], x['uuid'])
    #add branch points to the store:
    if len(newNodes) > 1:
        branchPoints.append(focusNode['uuid'])

def backtrack_from_branch():
    """ occasionally backtrack from a branch point: """
    if random() < BRANCH_BACKTRACK_AMNT and len(branchPoints) > 0:
        rndBranch = choice(branchPoints)
        rndBranchNode = allNodes[rndBranch]
        length_of_branch = rndBranchNode['distance_from_branch']
        branchPoint = BACKTRACK_STEP_NUM(length_of_branch)
        currentNodeUUID = rndBranch
        for x in range(branchPoint):
            currentNodeUUID = graph.predecessors(currentNodeUUID)[0]

        potentialNode = allNodes[currentNodeUUID]
        if potentialNode['remaining'] > 0 and len(frontier) < MAX_FRONTIER_NODES:
            potentialNode['perpendicular'] = True
            frontier.append(currentNodeUUID)


def get_branch_point(nodeUUID):
    """ skip down the successor chain until finding a branch """
    currentUUID = nodeUUID
    successors = graph.successors(currentUUID)
    while len(successors) == 1:
        currentUUID = successors[0]
        successors = graph.successors(currentUUID)
    return currentUUID

def get_path(nodeUUID):
    """ get all nodes from the current to the next branch """
    path = []
    successors = graph.successors(nodeUUID)
    while len(successors) == 1:
        path.append(successors[0])
        successors = graph.successors(path[-1])
    return path


            
##############################
# DRAWING
####################
def draw_hyphae():
    """ Simple draw routine, each node as a circle """
    logging.debug("Drawing")
    #clear the context
    utils.clear_canvas(ctx)
    #from the root node of the graph
    nodes = deque([x['uuid'] for x in root_nodes])
    #BFS the tree
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        #todo: get a line between currentNode and predecessor
        #draw the node / line
        ctx.set_source_rgba(*colours[currentNode['colour']])
        logging.debug("Circle: {:.2f},  {:.2f}".format(*currentNode['loc']))
        utils.drawCircle(ctx, *currentNode['loc'], currentNode['d']-SIZE_DIFF)
        #get it's children
        nodes.extend(graph.successors(currentUUID))
            
    return True

def draw_hyphae_2():
    """ Draw an alternate form of the graph """
    logging.debug("Drawing alternate")
    utils.clear_canvas(ctx)
    nodes = deque([graph.successors(x['uuid']) for x in root_nodes])
    #BFS the tree:
    ctx.set_source_rgba(*MAIN_COLOUR)
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        prev = allNodes[graph.predecessors(currentNode['uuid'])[0]]
        points = utils.createLine(*currentNode['loc'], *prev['loc'], LINE_DISTORTION_UPSCALING)
        length_of_line = np.linalg.norm(points[-1] - points[0])
        distorted = utils.displace_along_line(points, \
                                              length_of_line * LINE_PROPORTION_DISTORTION, \
                                              LINE_DISTORTION_UPSCALING)
        nodes.extend(graph.successors(currentUUID))
        for x, y in distorted:
            utils.drawCircle(ctx, x, y, MIN_NODE_SIZE)
        #for x, y in points:
        #    utils.drawCircle(ctx, x, y, \
        #    utils.clamp(currentNode['d']-SIZE_DIFF, MIN_NODE_SIZE, NODE_START_SIZE))

    return True
    
def draw_hyphae_3():
    """ An alternate draw routine, drawing lines for branches """    
    utils.clear_canvas(ctx)
    nodes = deque([x['uuid'] for x in root_nodes])
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        branchUUID = get_branch_point(currentUUID)
        branchNode = allNodes[branchUUID]
        
        ctx.set_source_rgba(*colours[currentNode['colour']])
        ctx.set_line_width(LINE_WIDTH)
        utils.drawCircle(ctx, *currentNode['loc'], currentNode['d']-SIZE_DIFF)
        ctx.move_to(*currentNode['loc'])
        ctx.line_to(*branchNode['loc'])
        ctx.stroke()
        nodes.extend(graph.successors(branchNode['uuid']))
        for succUUID in graph.successors(branchNode['uuid']):
            succNode = allNodes[succUUID]
            ctx.move_to(*branchNode['loc'])
            ctx.line_to(*succNode['loc'])
            ctx.stroke()

def draw_hyphae_4():
    """ Another alternate draw routine, which paths wiggles """
    utils.clear_canvas(ctx)
    nodes = deque([x['uuid'] for x in root_nodes])
    while len(nodes) > 0:
        currentUUID = nodes.popleft()
        currentNode = allNodes[currentUUID]
        pathUUIDs = get_path(currentUUID)

        ctx.set_source_rgba(*colours[currentNode['colour']])
        ctx.set_line_width(LINE_WIDTH)
        utils.drawCircle(ctx, *currentNode['loc'], currentNode['d']-SIZE_DIFF)

        if len(pathUUIDs) == 0:
            nodes.extend(graph.successors(currentUUID))
            for succUUID in graph.successors(currentUUID):
                lastNode = allNodes[currentUUID]
                succNode = allNodes[succUUID]
                ctx.move_to(*lastNode['loc'])
                ctx.line_to(*succNode['loc'])
                ctx.stroke()
            continue
        
        ctx.move_to(*currentNode['loc'])
        for nextUUID in pathUUIDs:
            nextNode = allNodes[nextUUID]
            ctx.line_to(*nextNode['loc'])
        ctx.stroke()
        nodes.extend(graph.successors(pathUUIDs[-1]))            

        for succUUID in graph.successors(pathUUIDs[-1]):
            lastNode = allNodes[pathUUIDs[-1]]
            succNode = allNodes[succUUID]
            ctx.move_to(*lastNode['loc'])
            ctx.line_to(*succNode['loc'])
            ctx.stroke()

##############################
#GROWING
####################
def grow(node=None):
    """ Grow a single node out """    
    logging.debug("Growing")
    #pick a frontier node
    if node is not None:
        focusNodeUUID = node
    else:
        focusNodeUUID = frontier.popleft()
    focusNode = allNodes[focusNodeUUID]
    newPoint = determine_new_point(focusNode)
    newPositions, decay, distance_from_branch = split_if_necessary(newPoint, focusNode)

    if positions_collide(newPositions, focusNode):
        return False

    grow_suitable_nodes(newPositions, decay, distance_from_branch, focusNode)
    backtrack_from_branch()
    
    return False
            
def grow_frontier():
    """ Grow every node in the frontier in a single step  """
    global frontier
    current_frontier = frontier
    frontier = deque()
    for node in current_frontier:
        grow(node)

    return False
            
def initialise():
    """ Setup the starting state of the growth """
    logging.info("Setting up root node")
    for i, loc in enumerate(START_NODES):
        root_nodes.append(create_node(loc, NODE_START_SIZE, colour=i))


##############################
if __name__ == "__main__":
    logging.info('Starting main')
    initialise()
    frame_index = 0
    growSaysFinish = False
    while not all_frontiers_are_at_boundary() and not growSaysFinish and frame_index < MAX_GROWTH_STEPS:
        frame_index += 1
        growSaysFinish = grow_frontier()
        logging.info(frame_index)
        if not ANIMATE:
            continue

        if frame_index % DRAW_STEP == 0:
            if DRAW_NODES:
                draw_hyphae()
            elif DRAW_LINES:
                draw_hyphae_3()
            elif DRAW_PATHS:
                draw_hyphae_4()
            else:
                draw_hyphae_2()
            utils.write_to_png(surface, join(OUTPUT_DIR, OUTPUT_FILE), frame_index)
        if frame_index % 50 == 0:
            logging.info("...")
    draw_hyphae()
    utils.write_to_png(surface, join(OUTPUT_DIR, OUTPUT_FILE), "FINAL")
    utils.pickle_graph(allNodes,graph,PICKLE_FILENAME,PICKLE_DIR)
