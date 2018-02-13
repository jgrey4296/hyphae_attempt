from .HyphaeDrawSuperClass import Hyphae_Draw
from collections import deque
from hyphae.constants import SIZE_DIFF, LINE_WIDTH, MAIN_COLOUR, LINE_DISTORTION_UPSCALING, LINE_PROPORTION_DISTORTION, MIN_NODE_SIZE
import cairo_utils as utils
import logging as root_logger
import numpy as np
import IPython
logging = root_logger.getLogger(__name__)

class NodePathsAlt(Hyphae_Draw):

    def __init__(self, hyphae_instance, N=5):
        super().__init__(hyphae_instance, N=N, filename="hyphae_nodepaths_alt")

    def draw(self):
        """ Draw an alternate form of the graph """
        logging.debug("Drawing alternate")
        utils.drawing.clear_canvas(self.ctx)
        nodes = deque([a for x in self.instance.root_nodes for a in self.instance.graph.successors(x.id)])
        #BFS the tree:
        self.ctx.set_source_rgba(*MAIN_COLOUR)
        while len(nodes) > 0:
            currentID = nodes.popleft()
            currentNode = self.instance.allNodes[currentID]
            prev = self.instance.allNodes[self.instance.graph.predecessors(currentNode.id)[0]]
            points = utils.drawing.createLine(*currentNode.loc, *prev.loc, LINE_DISTORTION_UPSCALING)
            length_of_line = np.linalg.norm(points[-1] - points[0])
            distorted = utils.math.displace_along_line(points,
                                                  length_of_line * LINE_PROPORTION_DISTORTION,
                                                  LINE_DISTORTION_UPSCALING)
            nodes.extend(self.instance.graph.successors(currentID))
            for x, y in distorted:
                utils.drawing.drawCircle(self.ctx, x, y, MIN_NODE_SIZE)
            #for x, y in points:
            #    utils.drawing.drawCircle(self.ctx, x, y, \
                #    utils.math.clamp(currentNode['d']-SIZE_DIFF, MIN_NODE_SIZE, NODE_START_SIZE))
