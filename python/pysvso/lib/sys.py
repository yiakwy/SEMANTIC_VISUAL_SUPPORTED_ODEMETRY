import os
import sys
import logging

def add_path(path):
    path = os.path.abspath(path)
    if path not in sys.path:
        logging.info("loading path %s ..." % path)
        sys.path.insert(0, path)
    else:
        logging.info("path %s exists!" % path)
