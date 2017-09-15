"""
An re-implementation of inconvergent's hyphae
"""
from collections import namedtuple, deque
from numpy.random import random
from os.path import join
from random import choice
from uuid import uuid1
import IPython
import cairo
import cairo_utils as utils
import logging as root_logger
import math
import networkx as nx
import numpy as np
import pyqtree

from .constants import *
from .Node import Node

logging = root_logger.getLogger(__name__)

class Hyphae:
    """ A reimplementation of Inconvergent's version of hyphae """
    
    def __init__(self, debug=False, debug_dir="output"):
        self.allNodes = {}
        self.branchPoints = []
        #Keep track of nodes in a directed graph
        self.graph = nx.DiGraph()
        #check for neighbours with the qtree
        self.qtree = pyqtree.Index(bbox=BOUNDS)
        #Qtree usage:
        #qtree.insert(item=item, bbox=item.bbox.flatten)
        #matches = qtree.intersect(bbox)

        self.frontier = deque()
        self.root_nodes = []
        #Colours for each node
        self.colours = [NODE_COLOUR() for x in START_NODES]

        self.debug_flag = debug
        self.debug_dir = debug_dir
        if self.debug_flag:
            #setup the debug drawing subclass
            1
            
            
        
    def create_node(self, location, d, distance=0, colour=0):
        """
        Create a node, add it to the graph, frontier, and quadtree
        """
        logging.debug("Creating a node for location: {}".format(location))
        if colour > len(self.colours):
            raise Exception("Tried to create a node with a non-existent colour")

        newNode = Node(location, d, distance, colour=colour)
        # newNode = {'loc':location, 'd':d, 'uuid': uuid1(), 'remaining':MAX_ATTEMPTS_PER_NODE,
        #            'distance_from_branch': distance, 'perpendicular' : False, 'colour' : colour}
        self.allNodes[newNode.id] = newNode
        self.graph.add_node(newNode.id)
        self.frontier.append(newNode.id)
        self.qtree.insert(item=newNode.id, bbox=newNode.bbox.flatten())
        return newNode

    def get_neighbourhood(self, x, y, d):
        """
        for a given new node, get nodes local to it spatially
        + the predecessor chain
        """
        delta = d*NEIGHBOUR_DELTA
        bbox = [x-delta, y-delta,
                x+delta, y+delta]
        logging.debug("Neighbourhood of ({},{}) -> bbox {}".format(x, y, bbox))
        matches = self.qtree.intersect(bbox)
        assert(all([x in self.allNodes for x in matches]))
        match_nodes = [self.allNodes[x] for x in matches]
        assert(len(matches) == len(match_nodes))
        return match_nodes

    def get_node_neighbourhood(self, node):
        assert(isinstance(node, Node))
        return self.get_neighbourhood(node.loc[0], node.loc[1], node.d)

    def filter_frontier_to_boundary(self):
        """ verify the distances of the frontier to the centre """
        logging.debug("Checking {} frontiers are at boundary".format(len(self.frontier)))
        #distance check all nodes in the frontier
        assert(all([x in self.allNodes for x in self.frontier]))
        nodesFromIDs = [self.allNodes[x] for x in self.frontier]
        distances = [x.distance_to(CENTRE) for x in nodesFromIDs]
        paired = zip(self.frontier, distances)
        logging.debug("Distances: {}".format(distances))
        #filter the frontier:
        self.frontier = deque([x for x, y in paired if y < HYPHAE_CIRC])
        return len(self.frontier) == 0

    def determine_new_point(self, node):
        """
        Given a node, calculate a next node location
        """
        assert(isinstance(node, Node))
        predecessorIDs = self.graph.predecessors(node.id)
        if node.perpendicular is True and len(predecessorIDs) > 0:
            #get the norm, rotate 90 degrees
            assert(all([x in self.allNodes for x in predecessorIDs]))
            predecessor = self.allNodes[predecessorIDs[0]]
            normalized = predecessor.get_normal(node)
            direction = choice([[[0, -1], [1, 0]], [[0, 1], [-1, 0]]])
            perpendicular = normalized.dot(direction)
            newPoint = node.loc + (perpendicular * (2*node.d))
        elif len(predecessorIDs) == 0:
            #no predecessor, pick a random direction
            logging.debug("No predecessor, picking random direction")
            #todo: rotate around the point
            newPoint = node.move()
        else:
            logging.debug("Extending vector")
            #create a vector out of the pair / Alt: move -> d(p, x) < d(n, x)
            assert(predecessorIDs[0] in self.allNodes)
            predecessor = self.allNodes[predecessorIDs[0]]
            normalized = predecessor.get_normal(node)
            newPoint = node.move(normalized)
            if random() < WIGGLE_CHANCE:
                newPoint = utils.rotatePoint(newPoint, node.loc, 
                                             radMin=-(WIGGLE_AMNT + WIGGLE_VARIANCE), 
                                             radMax=(WIGGLE_AMNT + WIGGLE_VARIANCE))

        return newPoint

    def split_if_necessary(self, point, focusNode):
        """ Split branch based on split chance """
        assert(isinstance(focusNode, Node))
        if not focusNode.perpendicular and random() < SPLIT_CHANCE:
            s1 = utils.rotatePoint(point, focusNode.loc,
                                   radMin=-(SPLIT_ANGLE+SPLIT_ANGLE_VARIANCE), 
                                   radMax=-(SPLIT_ANGLE-SPLIT_ANGLE_VARIANCE))
            s2 = utils.rotatePoint(point, focusNode.loc,
                                   radMin=SPLIT_ANGLE-SPLIT_ANGLE_VARIANCE, 
                                   radMax=SPLIT_ANGLE+SPLIT_ANGLE_VARIANCE)
            newPositions = [s1, s2]
            decay = NODE_SIZE_DECAY
            distance_from_branch = 0
        elif focusNode.perpendicular:
            newPositions = [point]
            decay = 0.0
            distance_from_branch = 0
        else:
            newPositions = [point]
            decay = 0.0
            distance_from_branch = focusNode.distance_from_branch + 1

        return (newPositions, decay, distance_from_branch)


    def positions_collide(self, positions, focusNode):
        """ 
        See if the positions specified are too close to any existing nodes
        """
        assert(isinstance(positions, list))
        assert(isinstance(focusNode, Node))
        predecessorIDs = self.graph.predecessors(focusNode.id)
        neighbours = [x for newPos in positions for x in self.get_neighbourhood(*newPos,
                                                                                focusNode.d) \
                      if x.id not in predecessorIDs and x.id != focusNode.id]
        
        if len(neighbours) != 0:
            logging.debug("There are {} intersections,  not adding a new node".format(len(neighbours)))
            focusNode.attempt()
            if focusNode.open() and len(self.frontier) < MAX_FRONTIER_NODES:
                self.frontier.append(focusNode.id)
            return True
   
        return False
  

    def grow_suitable_nodes(self, newPositions, decay, distance_from_branch, focusNode):
        """
        Create new nodes,  storing any branch points,  and linking the edge to its parent
        """
        #retrieve or create a colour:
        if focusNode.perpendicular:
            self.colours.append(NODE_COLOUR())
            colour_index = len(self.colours) - 1
        else:
            colour_index = focusNode.colour
        #create the nodes:
        newNodes = [self.create_node(x, focusNode.d - decay,
                                     distance=distance_from_branch, colour=colour_index) \
                    for x in newPositions]
        #add the nodes to the graph:
        for x in newNodes:
            self.graph.add_edge(focusNode.id, x.id)
        #add branch points to the store:
        if len(newNodes) > 1:
            self.branchPoints.append(focusNode.id)

    def backtrack_from_branch(self):
        """ occasionally backtrack from a branch point: """
        if random() < BRANCH_BACKTRACK_AMNT and bool(self.branchPoints):
            rndBranch = choice(self.branchPoints)
            assert(rndBranch in self.allNodes)
            rndBranchNode = self.allNodes[rndBranch]
            length_of_branch = rndBranchNode.distance_from_branch
            branchPoint = BACKTRACK_STEP_NUM(length_of_branch)
            currentNodeID = rndBranch
            for x in range(branchPoint):
                currentNodeID = self.graph.predecessors(currentNodeID)[0]

            assert(currentNodeID in self.allNodes)
            potentialNode = self.allNodes[currentNodeID]
            if potentialNode.open() and len(self.frontier) < MAX_FRONTIER_NODES:
                potentialNode.perpendicular = True
                self.frontier.append(currentNodeID)


    def get_branch_point(self, nodeID):
        """ skip down the successor chain until finding a branch """
        currentID = nodeID
        successors = self.graph.successors(currentID)
        while len(successors) == 1:
            currentID = successors[0]
            successors = self.graph.successors(currentID)
        return currentID

    def get_path(self, nodeID):
        """ get all nodes from the current to the next branch """
        path = []
        successors = self.graph.successors(nodeID)
        while len(successors) == 1:
            path.append(successors[0])
            successors = self.graph.successors(successors[0])
        return path

    def grow(self, node=None):
        """ Grow a single node out """    
        logging.debug("Growing")
        #pick a frontier node
        if isinstance(node, Node):
            focusNodeID = node.id
        elif node is not None:
            focusNodeID = node
        else:
            focusNodeID = self.frontier.popleft()
        assert(focusNodeID in self.allNodes)
        focusNode = self.allNodes[focusNodeID]
        newPoint = self.determine_new_point(focusNode)
        newPositions, decay, distance_from_branch = self.split_if_necessary(newPoint, focusNode)

        if not self.positions_collide(newPositions, focusNode):
            self.grow_suitable_nodes(newPositions, decay, distance_from_branch, focusNode)
            self.backtrack_from_branch()
            
    def grow_frontier(self):
        """ Grow every node in the frontier in a single step  """
        current_frontier = self.frontier
        self.frontier = deque()
        for node in current_frontier:
            self.grow(node)

            
    def initialise(self, start_nodes=START_NODES):
        """ Setup the starting state of the growth """
        logging.info("Setting up root node")
        for i, loc in enumerate(start_nodes):
            self.root_nodes.append(self.create_node(loc, NODE_START_SIZE, colour=i))

    def load(self):
        raise Exception("Not implemented")

    def save(self):
        raise Exception("Not implemented")
            

    def run(self, max_frame=10000):
        frame_index = 0
        
        while not self.filter_frontier_to_boundary() and frame_index < MAX_GROWTH_STEPS:
            self.grow_frontier()

            if self.debug_flag:
                #trigger debug drawing
                True
            frame_index += 1

        #finished:
        if self.debug_flag:
            #draw the final output
            1
