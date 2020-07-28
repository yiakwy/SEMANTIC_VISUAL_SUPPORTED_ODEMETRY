import sys
import os
import logging
logging.basicConfig(level=logging.INFO)

try:
    add_path
except:
    def add_path(path):
        path = os.path.abspath(path)
        if path not in sys.path:
            logging.info("loading path %s ..." % path)
            sys.path.insert(0, path)
        else:
            logging.info("path %s exists!" % path)

pwd = os.path.dirname(os.path.realpath(__file__))
add_path(pwd)

## Add Config to Python Path
add_path(os.path.join(pwd, '..', 'lib'))
add_path(os.path.join(pwd, '..', 'config'))

## Import models
from .sfe import *
from .vgg16 import *
