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

import numpy as np
import pandas
import networkx as nx
import cairocffi as cairo
import utils
from collections import namedtuple
import pyqtree

#CONSTANTS:
N = 8
X = pow(2,N)
Y = pow(2,N)
START = (0.5,0.5)
OUTPUT_FILE = "output"
#Keep track of nodes in a directed graph
logging.info("Setting up Graph and QuadTree")
graph = nx.DiGraph()
bounds = [0,0,1,1]
#check for neighbours with the qtree 
qtree = pyqtree.Index(bbox=bounds)
#qtree.insert(item=item, bbox=item.bbox)
#matches = qtree.intersect(bbox)

#CAIRO: --------------------
logging.info("Setting up Cairo")
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, X,Y)
ctx = cairo.Context(surface)
ctx.scale(X,Y) #coords in 0-1 range

#Utility functions:
def getNeighbourhood(node_coords):
    """
    for a given new node, get nodes local to it spatially
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


#Main Growth function:
def grow():
    return False


def draw_rect(ctx):
    logging.info(ctx.set_source_surface)
    ctx.set_source_rgba(0,1,1,0.5)
    utils.drawRect(ctx,0.1,0.1,0.9,0.)
    return False


if __name__ == "__main__":
    logging.info("Performing main")
    ctx.set_source_rgba(0,1,1,0.8)
    utils.drawRect(ctx,0.2,0.2,0.8,0.8)
    utils.write_to_png(surface,OUTPUT_FILE)
