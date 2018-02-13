from .HyphaeDrawSuperClass import Hyphae_Draw
from collections import deque
from hyphae.constants import SIZE_DIFF, LINE_WIDTH
import cairo_utils as utils
import logging as root_logger

logging = root_logger.getLogger(__name__)

class NodePaths(Hyphae_Draw):

    def __init__(self, hyphae_instance, N=5):
        super().__init__(hyphae_instance, N=N, filename="hyphae_nodepaths")

    def draw(self):
        """ Simple draw routine, each node as a circle """
        logging.debug("Drawing")
        #from the root node of the graph
        nodes = deque([x for x in self.instance.frontier])
        #BFS the tree
        while len(nodes) > 0:
            currentID = nodes.popleft()
            currentNode = self.instance.allNodes[currentID]
            #todo: get a line between currentNode and predecessor
            #draw the node / line
            self.ctx.set_source_rgba(*currentNode.colour)
            logging.debug("Circle: {:.2f},  {:.2f}".format(*currentNode.loc))
            utils.drawing.drawCircle(self.ctx, *currentNode.loc, currentNode.d - SIZE_DIFF)
            #get it's children
            #nodes.extend(self.instance.graph.successors(currentID))
