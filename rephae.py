"""
    Reloads a pickled graph for modification and re-drawing
"""
# Setup root_logger:
import logging as root_logger
LOGLEVEL = root_logger.DEBUG
LOG_FILE_NAME = "rephae.log"
root_logger.basicConfig(filename=LOG_FILE_NAME, level=LOGLEVEL, filemode='w')

console = root_logger.StreamHandler()
console.setLevel(root_logger.INFO)
root_logger.getLogger('').addHandler(console)
logging = root_logger.getLogger(__name__)
##############################
# IMPORTS
####################
import utils
import networkx as nx
import numpy as np
import cairo


##############################
# CONSTANTS
####################
PICKLE_DIR = "output_pickle"
PICKLE_NAME = "hyphae_graph"
OUTPUT_DIR = "output"
OUTPUT_FILE = "rephae"

N = 10
X = pow(2,N)
Y = pow(2,N)
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, X,Y)
ctx = cairo.Context(surface)
ctx.scale(X,Y) #coords in 0-1 range

##############################
# VARIABLES
####################
allNodes, graph = utils.load_pickled_graph(PICKLE_NAME,PICKLE_DIR)

##############################
# Utilities
####################

##############################
# Core Functions
####################

########################################
if __name__ == "__main__":
    logging.info("Starting ")
    logging.info("Allnodes       : {}".format(len(allNodes)))
    logging.info("Graph nodes    : {}".format(len(graph.nodes())))
    logging.info("Graph edges    : {}".format(len(graph.edges())))
    
                 

