from .HyphaeDrawSuperClass import Hyphae_Draw
from collections import deque
from hyphae.constants import SIZE_DIFF, LINE_WIDTH
import cairo_utils as utils
import logging as root_logger
logging = root_logger.getLogger(__name__)

class LinePaths(Hyphae_Draw):

    def __init__(self, hyphae_instance, N=5):
        super().__init__(hyphae_instance, N=N, filename="hyphae_straightpaths")

    def draw(self):
        """ Another alternate draw routine, which paths wiggles """
        utils.clear_canvas(self.ctx)
        nodes = deque([x.id for x in self.instance.root_nodes])
        while len(nodes) > 0:
            currentID = nodes.popleft()
            currentNode = self.instance.allNodes[currentID]
            pathIDs = self.instance.get_path(currentID)

            self.ctx.set_source_rgba(*currentNode.colour)
            self.ctx.set_line_width(LINE_WIDTH)
            utils.drawCircle(self.ctx, *currentNode.loc, currentNode.d - SIZE_DIFF)

            if len(pathIDs) == 0:
                successors = self.instance.graph.successors(currentID)
                nodes.extend(successors)
                for succID in successors:
                    lastNode = self.instance.allNodes[currentID]
                    succNode = self.instance.allNodes[succID]
                    self.ctx.move_to(*lastNode.loc)
                    self.ctx.line_to(*succNode.loc)
                    self.ctx.stroke()
                continue
        
            self.ctx.move_to(*currentNode.loc)
            for nextID in pathIDs:
                nextNode = self.instance.allNodes[nextID]
                self.ctx.line_to(*nextNode.loc)
            self.ctx.stroke()
            
            nodes.extend(self.instance.graph.successors(pathIDs[-1]))            
            for succID in self.instance.graph.successors(pathIDs[-1]):
                lastNode = self.instance.allNodes[pathIDs[-1]]
                succNode = self.instance.allNodes[succID]
                self.ctx.move_to(*lastNode.loc)
                self.ctx.line_to(*succNode.loc)
                self.ctx.stroke()
