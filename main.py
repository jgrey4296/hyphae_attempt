"""
    Hyphae main

"""
##############################
# IMPORTS
####################
# Setup root_logger:
import logging as root_logger
import hyphae

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

DEBUG=False

########################################
if __name__ == "__main__":
    logging.info("Starting ")
    hyphae_instance = hyphae(debug=DEBUG)
    hyphae.initialise()
