from .HyphaeDrawSuperClass import Hyphae_Draw
from collections import deque
from hyphae.constants import SIZE_DIFF, LINE_WIDTH
import cairo_utils as utils
import logging as root_logger
logging = root_logger.getLogger(__name__)


class StraightPaths(Hyphae_Draw):

    def __init__(self, hyphae_instance, N=5):
        super().__init__(hyphae_instance, N=N, filename="hyphae_lines")
        
    def draw(self):
        """ An alternate draw routine, drawing lines for branches """    
        utils.drawing.clear_canvas(self.ctx)
        nodes = deque([x.id for x in self.instance.root_nodes])
        while len(nodes) > 0:
            currentID = nodes.popleft()
            currentNode = self.instance.allNodes[currentID]
            branchID = self.instance.get_branch_point(currentID)
            branchNode = self.instance.allNodes[branchID]
        
            self.ctx.set_source_rgba(*currentNode.colour)
            self.ctx.set_line_width(LINE_WIDTH)
            utils.drawing.drawCircle(self.ctx, *currentNode.loc, currentNode.d - SIZE_DIFF)
            self.ctx.move_to(*currentNode.loc)
            self.ctx.line_to(*branchNode.loc)
            self.ctx.stroke()
            nodes.extend(self.instance.graph.successors(branchNode.id))
            for succID in self.instance.graph.successors(branchNode.id):
                succNode = self.instance.allNodes[succID]
                self.ctx.move_to(*branchNode.loc)
                self.ctx.line_to(*succNode.loc)
                self.ctx.stroke()
