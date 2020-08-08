import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
Project_base = os.path.join(ROOT, "../..")

# data set
DATA_DIR = os.path.abspath(os.path.join(Project_base, "data"))

# output
OUTPUT_DIR = os.path.abspath(os.path.join(Project_base, "log"))

# pretrained models
# self trained


# thirdparty models implementation
MRCNN = os.path.abspath(os.path.join(Project_base, "vendors/github.com/Mask_RCNN"))

# coco
MRCNN_COCO_DATASET = os.path.join(MRCNN, "samples/coco")
COCO_MODEL_PATH = os.path.join(DATA_DIR, "models/coco", "mask_rcnn_coco.h5")
MODEL_PATH = os.path.join(OUTPUT_DIR, "models")

# images
task = "tum"
data = ""
seqid = "rgbd_dataset_freiburg1_xyz/rgb"
# Monocular camera task does not hold this variable
cameraid = ""

# see scripts/compose_tum_video.sh for details
IMAGE_DIR = os.path.join(DATA_DIR, "{task}/${data}/{seqid}/{cameraid}".format(
    task=task,
    data=data,
    seqid=seqid,
    cameraid=cameraid
))