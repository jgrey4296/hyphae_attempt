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
#import cairocffi as cairo
import utils
from collections import namedtuple
import pyqtree
import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk


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
# logging.info("Setting up Cairo")
# surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, X,Y)
# ctx = cairo.Context(surface)
# ctx.scale(X,Y) #coords in 0-1 range

#GTK: --------------------

#Utility functions:
def write_image_and_exit(*args):
    utils.write_to_png(surface,OUTPUT_FILE)
    Gtk.main_quit(*args)
               
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
    logging.info("Seting up Gtk")
    window = Gtk.Window()
    window.resize(X,Y)
    subwin = window.get_window()
    cr = subwin.cairo_create()
    #darea = Gtk.DrawingArea()
    #window.add(darea)
    #window.connect("destroy",write_image_and_exit)
    window.connect('draw',draw_rect)
    window.show_all()
    #darea.connect('draw',draw_rect)
    Gtk.main()
    
