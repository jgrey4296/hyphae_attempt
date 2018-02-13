from os.path import join
import cairo_utils as utils
from hyphae.constants import OUTPUT_DIR

class Hyphae_Draw:
    """ Superclass for Hyphae Debug Drawing  """
    
    def __init__(self, hyphae_instance, N=5, filename="hyphae_debug"):
        assert(hyphae_instance is not None)
        self.instance = hyphae_instance
        #setup the surface, ctx, size etc
        surf, ctx, size, n = utils.drawing.setup_cairo(N=N)
        self.surface = surf
        self.ctx = ctx
        self.size = size
        self.n = n
        self.filename = filename

    def draw(self):
        raise Exception("Hyphae_Draw.draw is Abstract")


    def write_file(self, i=None):
        utils.drawing.write_to_png(self.surface, join(OUTPUT_DIR, self.filename), i=i)
