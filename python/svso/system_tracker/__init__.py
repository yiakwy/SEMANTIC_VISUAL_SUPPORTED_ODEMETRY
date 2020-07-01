import sys
import os
import logging

try:
    add_path
except NameError:
    def add_path(path):
        path = os.path.abspath(path)
        if path not in sys.path:
            logging.info("loading path %s ..." % path)
            sys.path.insert(0, path)
        else:
            logging.info("path %s exists!" % path)

pwd = os.path.dirname(os.path.realpath(__file__))
add_path(os.path.join(pwd, '../../../', 'build/proto_codec'))