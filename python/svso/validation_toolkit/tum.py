# reference EVO toolkit
from evo.tools import file_interface
from evo.tools.file_interface import csv_read_matrix, FileInterfaceException
from svso.lib.maths.rotation import Euler, Quaternion

import numpy as np

import logging
from svso.lib.log import LoggerAdaptor

_logger = logging.getLogger("validation.tum")

from svso.config import Settings

settings = Settings()

TUM_DATASET_NAME = settings.DATASET_NAME # "rgbd_dataset_freiburg1_xyz"
HDD  = settings.HDD # "/home/yiakwy"
ROOT = settings.ROOT # "{hdd}/WorkSpace".format(hdd=HDD)
REPO = settings.REPO # "SEMANTIC_SLAM"
PROJECT_ROOT = settings.PROJECT_ROOT # "{root}/Github/{repo}".format(root=ROOT, repo=REPO)
# TUM_DATA_DIR =  "{project_base}/data/tum/{dataset_name}".format(project_base=Project_base,
#                                                                dataset_name=TUM_DATASET_NAME)
TUM_DATA_DIR = settings.DATA_DIR

Project_base = PROJECT_ROOT

class Trajectory3D:
    # Implements STL iterator
    class Trajectory3DIterator(object):
        def __init__(self, trajectory):
            self._trajectory = trajectory
            self.counter = self.__counter__()

        def __iter__(self):
            return self

        def __counter__(self):
            l = len(self._trajectory)
            # one dimension index
            ind = 0
            while True:
                yield ind
                ind += 1
                if ind >= l:
                    break

        def __next__(self):
            try:
                ind = next(self.counter)
                return self._trajectory[ind]
            except StopIteration:
                raise StopIteration()

        def __str__(self):
            return "Trajectory iterator"

    def __init__(self, timestamp, rots, trans):

        self.timestamps = timestamp

        # Rotations
        self.rots = rots

        # Translations
        self.trans = trans
        pass

    def __iter__(self):
        data = np.c_[self.timestamps, self.trans, self.rots]
        return Trajectory3D.Trajectory3DIterator(data)

    # @todo TODO return aligned pose data
    # This parts implements EVO synchronoization algorithm to fetch ground truth for a picture timestamp specified
    def query_timestamp(self, timestamp, max_diff=0.01):
        diffs = np.abs(self.timestamps - timestamp)
        ind = np.argmin(diffs)
        if diffs[ind] > max_diff:
            print("ind: %d, diff[ind]: %f" % (ind, diffs[ind]))
            raise Exception("There is no valid matched pose for timestamp %f" % timestamp)
        data = np.c_[self.timestamps, self.trans, self.rots]
        return data[ind]
        pass


# reference https://github.com/MichaelGrupp/evo/blob/master/evo/tools/file_interface.py
# refactor for runtime GT usage
def read_tum_trajectory_file(file_path):
    # tum ground truth trajectory format:
    #  timestamp tx ty tz qx qy qz qw

    # internally call cvs module to parse the file
    raw_mat = csv_read_matrix(file_path, delim=" ", comment_str="#")
    if len(raw_mat) > 0 and len(raw_mat[0]) != 8:
        raise FileInterfaceException("Not valid TUM trajectory file!")

    try:
        trajectories = np.array(raw_mat).astype(float)
    except ValueError:
        raise FileInterfaceException("Trajectory value must be type of float!")

    stamps = trajectories[:, 0]
    # translation
    xyz = trajectories[:, 1:4]
    # rotation in form of quaternion (qx, qy, qz, w)
    quat = trajectories[:, 4:]
    # shift component w in front column to be used by evo/core/transformation.py
    quat = np.roll(quat, 1, axis=1)
    if not hasattr(file_path, 'read'):
        print("Reading %d poses" % stamps.shape[0])
    return Trajectory3D(stamps, quat, xyz)
    pass


def read_tum_images_timestamp(file_path):
    # internally call cvs module to parse the file
    raw_mat = csv_read_matrix(file_path, delim=" ", comment_str="#")
    if len(raw_mat) > 0 and len(raw_mat[0]) != 2:
        raise FileInterfaceException("Not valid TUM images timestamp file!")
    if not hasattr(file_path, 'read'):
        print("Reading %d timestamps" % len(raw_mat))
    return raw_mat


# just for demo purpose
# @todo : TODO improve the algorithm using TopK (miniHeap) algorithm
def query_depth_img(depth_images, timestamp, max_diff=0.01):
    dataset = np.array(depth_images)
    stamps = dataset[:, 0].astype(float)
    diffs = np.abs(stamps - timestamp)
    ind = np.argmin(diffs)
    if diffs[ind] > max_diff:
        print("ind: %d, diff[ind]: %f" % (ind, diffs[ind]))
        print("There is no valid matched depth image for timestamp %f" % timestamp)
        return None

    depth_img = dataset[ind, 1]
    return depth_img


def Program(*raw_args):
    import os

    trajectories = read_tum_trajectory_file(os.path.join(TUM_DATA_DIR, "groundtruth.txt"))
    timestamps = read_tum_images_timestamp(os.path.join(TUM_DATA_DIR, "rgb.txt"))

    for i in range(3):
        timestamp = float(timestamps[i][0])
        mtched = trajectories.query_timestamp(timestamp, max_diff=0.01)
        t = np.array(mtched[1:4]).reshape((3, 1))
        qut = Quaternion(*mtched[4:])
        R = qut.Matrix4()
        print("======================================================")
        print("T %f matched trajectory data -> T %f: \nR:\n%s\nt:\n%s\n" % (
            timestamp,
            mtched[0],
            R,
            t
        ))

    depth_images = read_tum_images_timestamp(os.path.join(TUM_DATA_DIR, "depth.txt"))

    for i in range(3):
        timestamp = float(timestamps[i][0])
        depth_img = query_depth_img(depth_images, timestamp, max_diff=0.5)
        print("%s" % depth_img)

    return trajectories, timestamps, depth_images
    pass