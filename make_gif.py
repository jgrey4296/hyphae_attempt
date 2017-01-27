# Setup root_logger:
import logging as root_logger
LOGLEVEL = root_logger.DEBUG
logFileName = "gif_write.log"
root_logger.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')

console = root_logger.StreamHandler()
console.setLevel(root_logger.INFO)
root_logger.getLogger('').addHandler(console)
logging = root_logger.getLogger(__name__)

import sys
from os.path import isfile,exists,join, getmtime, splitext
from os import listdir
from PIL import Image,ImageSequence
import imageio
import re

GIF_OUTPUT = "."
GIF_NAME = "anim.gif"

OUTPUT_DIR = 'output'
FILE_FORMAT = '.png'
FPS=12
numRegex = re.compile(r'(\d+)')

def getNum(s):
    logging.info("Getting num of: {}".format(s))
    try:
        return int(numRegex.search(s).group(0))
    except Exception:
        return 9999999



files = [x for x in listdir(OUTPUT_DIR) if isfile(join(OUTPUT_DIR,x))]
files.sort(key=lambda x: getNum(x))

images = []

for filename in files:
    logging.info("Loading: {}".format(filename))
    images.append(imageio.imread(join(OUTPUT_DIR,filename)))

imageio.mimsave(join(GIF_OUTPUT,GIF_NAME),images,'GIF-FI')

# with imageio.get_writer(join(GIF_OUTPUT,GIF_NAME), mode='I') as writer:
#     for filename in files:
#         image = imageio.imread(join(OUTPUT_DIR,filename))
#         writer.append_data(image)

