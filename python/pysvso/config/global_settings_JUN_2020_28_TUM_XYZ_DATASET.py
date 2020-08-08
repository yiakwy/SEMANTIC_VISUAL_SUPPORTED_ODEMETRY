import sys
import os

# program control
DEBUG = True
USE_POSE_GROUND_TRUTH = True

# directories
HDD = "/home/yiakwy"
ROOT = "{hdd}/WorkSpace".format(hdd=HDD)
REPO = "SEMANTIC_SLAM"
PROJECT_ROOT = "{root}/Github/{repo}".format(root=ROOT, repo=REPO)

# video source
DATASET_NAME = "rgbd_dataset_freiburg1_xyz"
TUM_DATA_DIR = "{project_base}/data/tum/{dataset_name}".format(project_base=PROJECT_ROOT,
                                                               dataset_name=DATASET_NAME)

# camera
CAMERA_DEVICE="{project_base}/data/tum/camera1.yaml".format(project_base=PROJECT_ROOT)

# damp destination
VIDEO_DIR = "{project_base}/log/video".format(project_base=PROJECT_ROOT)
OUTPUT = VIDEO_DIR
SAVER = os.path.join(OUTPUT, "saver")
VIDEO_NAME = "freiburg1_xyz.mp4"