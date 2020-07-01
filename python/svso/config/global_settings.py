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

VIDEO_DIR = "{project_base}/log/video".format(project_base=PROJECT_ROOT)
OUTPUT = VIDEO_DIR
SAVER = os.path.join(OUTPUT, "saver")
VIDEO_NAME = "freiburg1_xyz.mp4"