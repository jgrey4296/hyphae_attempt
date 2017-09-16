"""
    Hyphae main

"""
##############################
# IMPORTS
####################
# Setup root_logger:
import logging as root_logger
import numpy as np
from hyphae import Hyphae
from hyphae.drawing import StraightPaths, LinePaths, NodePaths, NodePathsAlt
from hyphae import constants
from os.path import join, isfile, exists, isdir, splitext, expanduser
import sys
from cairo_utils.make_gif import Make_Gif
#Log Setup
LOGLEVEL = root_logger.DEBUG
LOG_FILE_NAME = "log.hyphae"
root_logger.basicConfig(filename=LOG_FILE_NAME, level=LOGLEVEL, filemode='w')

console = root_logger.StreamHandler()
console.setLevel(root_logger.INFO)
root_logger.getLogger('').addHandler(console)
logging = root_logger.getLogger(__name__)
##############################
# CONSTANTS
####################

DEBUG=True
DRAW_CLASS = LinePaths #StraightPaths, LinePaths, NodePaths, NodePathsAlt
N=10

MAKE_GIF = '-gif' in sys.argv

########################################
if __name__ == "__main__":
    logging.info("Starting ")
    hyphae_instance = Hyphae(debug=DEBUG, draw_class=DRAW_CLASS, N=N)
    hyphae_instance.initialise()
    hyphae_instance.run()
    hyphae_instance.save()
    if MAKE_GIF:
        maker = Make_Gif(source_dir="imgs")
        maker.run()

    
