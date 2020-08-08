import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))

# dataset
DATA_DIR = os.path.abspath(os.path.join(ROOT, "../../",  "log"))

# thirdparty models implementation
MRCNN = os.path.abspath(os.path.join("/home/yiakwy/WorkSpace/Github/Mask_RCNN"))