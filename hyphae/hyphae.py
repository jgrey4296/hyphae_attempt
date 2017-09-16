"""
An re-implementation of inconvergent's hyphae
"""
from collections import deque
from numpy.random import random
from random import choice
from uuid import uuid1
import cairo_utils as utils
from cairo_utils import pickles
import logging as root_logger
import math
import networkx as nx
import numpy as np
import pyqtree
import IPython
from . import constants
from .Node import Node

logging = root_logger.getLogger(__name__)

class Hyphae:
    """ A reimplementation of Inconvergent's version of hyphae """
    
    def __init__(self, debug=False, debug_dir="output", draw_class=None, N=5):
        self.allNodes = {}
        self.branchPoints = []
        #Keep track of nodes in a directed graph
        self.graph = nx.DiGraph()
        #check for neighbours with the qtree
        self.qtree = pyqtree.Index(bbox=constants.BOUNDS)
        #Qtree usage:
        #qtree.insert(item=item, bbox=item.bbox.flatten)
        #matches = qtree.intersect(bbox)

        self.frontier = deque()
        self.root_nodes = []

        self.debug_flag = debug
        self.debug_dir = debug_dir
        if self.debug_flag:
            assert(draw_class is not None)
            self.draw_instance = draw_class(self, N=N)

    def initialise(self, start_nodes=constants.START_NODES):
        """ Setup the starting state of the growth """
        logging.info("Setting up root nodes: {}".format(constants.START_NODES))
        for i, loc in enumerate(start_nodes):
            self.root_nodes.append(self.create_node(loc, constants.NODE_START_SIZE))

    def load(self):
        nodes, graph = pickles.load_pickled_graph("hyphae")
        self.allNodes = nodes
        self.graph = graph

    def save(self):
        pickles.pickle_graph(self.allNodes, self.graph, "hyphae")

        
        
    def create_node(self, location, d, distance=0, priorNode=None):
        """
        Create a node, add it to the graph, frontier, and quadtree
        """
        logging.debug("Creating a node for location: {}".format(location))

        if priorNode is not None:
            newNode = priorNode.mutate(location, d, distance,
                                       mod_chances=constants.mod_chances)
        else:
            newNode = Node(location, d, distance)
        # newNode = {'loc':location, 'd':d, 'uuid': uuid1(), 'remaining':MAX_ATTEMPTS_PER_NODE,
        #            'distance_from_branch': distance, 'perpendicular' : False, 'colour' : colour}
        self.allNodes[newNode.id] = newNode
        self.graph.add_node(newNode.id)
        self.frontier.append(newNode.id)
        self.qtree.insert(item=newNode.id, bbox=newNode.bbox.flatten())
        return newNode

    def get_node_neighbourhood(self, node, loc=None):
        assert(isinstance(node, Node))
        if loc is None:
            loc = node.loc
        bbox_raw = node.get_delta_bbox()
        bbox = np.array([loc - bbox_raw,
                         loc + bbox_raw])
        logging.debug("Neighbourhood of ({},{}) -> bbox {}".format(loc[0], loc[1], bbox))
        matches = self.qtree.intersect(bbox.flatten())
        assert(all([x in self.allNodes for x in matches]))
        match_nodes = [self.allNodes[x] for x in matches]
        assert(len(matches) == len(match_nodes))
        return match_nodes

    def filter_frontier_to_boundary(self):
        """ verify the distances of the frontier to the centre """
        logging.debug("Checking {} frontiers are at boundary".format(len(self.frontier)))
        #distance check all nodes in the frontier
        assert(all([x in self.allNodes for x in self.frontier]))
        nodesFromIDs = [self.allNodes[x] for x in self.frontier]
        distances = [x.distance_to(constants.CENTRE) for x in nodesFromIDs]
        paired = zip(self.frontier, distances)
        logging.debug("Distances: {}".format(distances))
        #filter the frontier:
        self.frontier = deque([x for x, y in paired if y < constants.HYPHAE_CIRC])
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
            if random() < node.wiggle_chance:
                newPoint = utils.rotatePoint(newPoint, node.loc, 
                                             radMin=-(node.wiggle_amnt + node.wiggle_variance), 
                                             radMax=(node.wiggle_amnt + node.wiggle_variance))

        return newPoint

    def split_if_necessary(self, point, focusNode):
        """ Split branch based on split chance """
        assert(isinstance(focusNode, Node))
        if not focusNode.perpendicular and random() < focusNode.split_chance:
            #branch
            s1 = utils.rotatePoint(point, focusNode.loc,
                                   radMin=-(focusNode.split_angle + focusNode.split_variance), 
                                   radMax=-(focusNode.split_angle - focusNode.split_variance))
            s2 = utils.rotatePoint(point, focusNode.loc,
                                   radMin=focusNode.split_angle - focusNode.split_variance, 
                                   radMax=focusNode.split_angle + focusNode.split_variance)
            #todo: figure out whats going on here
            if len(s1.shape) == 2:
                s1 = choice(s1)
            if len(s2.shape) == 2:
                s2 = choice(s2)
            newPositions = [s1, s2]
            decay = focusNode.size_decay
            distance_from_branch = 0
        elif focusNode.perpendicular:
            #go perpendicular, with a new branch
            newPositions = [point]
            decay = 0.0
            distance_from_branch = 0
        else:
            #extend the new branch
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
        bbox_delta = focusNode.d + focusNode.delta
        for pos in positions:
            neighbours = [x for newPos in positions for x in self.get_node_neighbourhood(focusNode,
                                                                                         loc=pos)
                          if x.id is not focusNode.id]
            fNodeDist = focusNode.distance_to(pos)
            distances = [x.distance_to(pos) for x in neighbours]
            too_close = [x for x in distances if x < fNodeDist or x < bbox_delta]
            if bool(too_close):
                logging.debug("There are {} collision,  not adding a new node".format(len(neighbours)))
                focusNode.attempt()
                if focusNode.open() and len(self.frontier) < constants.MAX_FRONTIER_NODES:
                    self.frontier.append(focusNode.id)
                return True

            distances_to_other_new_pos = [utils.get_distance(pos,x) for x in positions if all(x != pos)]
            if any([x < fNodeDist for x in distances_to_other_new_pos]):
                return True
        
            
        return False
  

    def grow_suitable_nodes(self, newPositions, decay, distance_from_branch, focusNode):
        """
        Create new nodes,  storing any branch points,  and linking the edge to its parent
        """
        #create the nodes:
        newNodes = [self.create_node(x, focusNode.d - decay,
                                     distance=distance_from_branch, priorNode=focusNode) \
                    for x in newPositions]
        #add the nodes to the graph:
        for x in newNodes:
            self.graph.add_edge(focusNode.id, x.id)
        #add branch points to the store:
        if len(newNodes) > 1:
            self.branchPoints.append(focusNode.id)

    def backtrack_from_branch(self):
        """ occasionally backtrack from a branch point: """
        if random() < constants.BRANCH_BACKTRACK_AMNT and bool(self.branchPoints):
            rndBranch = choice(self.branchPoints)
            assert(rndBranch in self.allNodes)
            rndBranchNode = self.allNodes[rndBranch]
            length_of_branch = rndBranchNode.distance_from_branch
            if length_of_branch < 2:
                return
            branchPoint = constants.BACKTRACK_STEP_NUM(length_of_branch)
            currentNodeID = rndBranch
            for x in range(branchPoint):
                currentNodeID = self.graph.predecessors(currentNodeID)[0]

            assert(currentNodeID in self.allNodes)
            potentialNode = self.allNodes[currentNodeID]
            if potentialNode.open() and len(self.frontier) < constants.MAX_FRONTIER_NODES:
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
        elif focusNode.open():
            focusNode.attempt()
            self.frontier.append(focusNodeID)

    def backtrack_random(self):
        if random() < constants.BRANCH_BACKTRACK_AMNT:
            randNode = choice(list(self.allNodes.values()))
            if randNode.open() and random() < randNode.backtrack_likelihood:
                randNode.perpendicular = True
                self.frontier.append(randNode.id)
            
            
    def grow_frontier(self):
        """ Grow every node in the frontier in a single step  """
        current_frontier = self.frontier
        self.frontier = deque()
        for node in current_frontier:
            self.grow(node)
            self.backtrack_from_branch()
            self.backtrack_random()


            

    def run(self, max_frames=constants.MAX_GROWTH_STEPS):
        logging.info("Running Calculation")
        frame_index = 0
        
        while not self.filter_frontier_to_boundary() and frame_index < max_frames:
            if self.debug_flag:
                self.draw_instance.draw()
                self.draw_instance.write_file(frame_index)

            self.grow_frontier()
            frame_index += 1

        #finished:
        if self.debug_flag:
            logging.info("Drawing Debug Final")
            self.draw_instance.draw()
            self.draw_instance.write_file()
