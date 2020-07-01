import cv2
import numpy as np
from svso.py_pose_graph.point3d import Point3D
from svso.predictors.ekf import BBoxKalmanFilter
from svso.predictors.opticalflow import OpticalFlowBBoxPredictor
from svso.lib.misc import AtomicCounter
import uuid
from svso.config import Settings

settings = Settings()

# setting debug variable
DEBUG = settings.DEBUG
USE_POSE_GROUND_TRUTH = settings.USE_POSE_GROUND_TRUTH
import logging

class AABB:

    def __init__(self, bottomLeft, topRight):
        self.bottomLeft = bottomLeft
        self.topRight = topRight
        self.centre = None
        self.width = None
        self.height = None

        self.Init()

    def Init(self):
        x1, y1, z1 = self.bottomLeft.data
        x2, y2, z2 = self.topRight.data

        self.centre = Point3D(x1 + x2 / 2., y1 + y2 / 2., z1 + z2 / 2.)

    def topCentre(self):
        x2, y2, z2 = self.topRight.data
        x3, y3, z3 = self.centre.data

        _topCentre = Point3D(x3, y3, z2)
        return _topCentre

    def toArray(self):
        x1, y1, z1 = self.bottomLeft.data
        x2, y2, z2 = self.topRight.data

        return np.array([
            [x1, y1, z1],
            [x2, y2, z2]
        ])


class Observation:

    Seq = AtomicCounter()

    def __init__(self, label, roi_kps, roi_features):
        # label
        self.label = label
        # score
        self.score = roi_features['score']
        # roi keypoints
        self.roi_kps = roi_kps
        # roi feature
        self.roi_features = roi_features
        # associated frame
        self.frame = None

        ## Updated while tracking ...

        #
        self.projected_pos = roi_features['box']
        #
        self.projected_mask = roi_features['mask']

        # key points used to reconstruct struction in 3d world
        self.points = {}

        #
        self.bbox = roi_features['box']

        ## Covisibility Graph Topology
        # recording upgraded 3dpoints for each ROI
        self.observations = {}

        ## Identity information

        self.seq = self.Seq()
        self.id = None
        self.uuid = uuid.uuid4()
        # computed key used to uniquely identify a key point
        self.key = None

        # parameters used for tracking
        self.kf = None
        # self.opticalPredictor = None
        self.predicted_states = self.projected_pos  # None

        # union set, here i uses Uncompressed Union Set to present films of the object
        self.parent = None
        self.records = []

        # rendering attributes
        self.color = None

        # AABB
        self._aabb = None

        # OBB
        self._Obb = None
        self._major = None
        self._minor = None
        self._W = None

        # Centroid
        self._centroid = None

    def Init(self):
        self.kf = BBoxKalmanFilter()
        # self.opticalPredictor = OpticalFlowBBoxPredictor()
        self.records.append((
            None,  # Observation
            self.frame.seq if hasattr(self, "frame") else -1,  # frameId
        ))
        return self

    def clone(self):
        ob = Observation(self.label, self.roi_kps, self.roi_features)

        ob.set_FromFrame(self.frame)

        for key, val in self.observations.items():
            ob.observations[key] = val

        #
        for k, p in self.points.items():
            ob.points[k] = p

        ob.Init()
        ob.update(self)
        ob.predicted_states = ob.projected_pos

        ob.color = self.color
        return ob

    # remove points based on depth distribution
    def culling(self):
        points = []
        for k, p in self.points.items():
            points.append(p)

        points = sorted(points, key=lambda p: p.z)

        # remove dpeth %5 smallest, and 95% biggest points
        l = len(points)
        if l == 0:
            return

        # outliers check
        median = points[int(l / 2)].z
        #
        ustd = np.std(np.array([p.z for p in points]))
        #
        if l < 10:
            if l < 4:
                # at least we need 4 points
                return
            left_idx = 0
            while left_idx < l:
                p = points[left_idx]
                if np.abs(p.z - median) > 1.2 * ustd:
                    left_idx += 1
                else:
                    break

            right_idx = l - 1
            while right_idx > left_idx:
                p = points[right_idx]
                if np.abs(p.z - median) > 1.2 * ustd:
                    right_idx -= 1
                else:
                    break

            # not modified
            if left_idx == 0 and right_idx == l - 1:
                return
        else:
            left_idx = int(l * 0.05)
            right_idx = int(l * 0.95)

            lower_bound = left_idx
            left_idx = 0
            while left_idx < lower_bound:
                p = points[left_idx]
                if np.abs(p.z - median) > 1.5 * ustd:
                    left_idx += 1
                else:
                    break

            upper_bound = right_idx
            right_idx = l - 1
            while right_idx > upper_bound:
                p = points[right_idx]
                if np.abs(p.z - median) > 1.5 * ustd:
                    right_idx -= 1
                else:
                    break
            pass

        print("[%s] culling points [0,%d], [%d,%d)" % (self, left_idx, right_idx, l))

        #
        for i in range(left_idx + 1):
            p = points[i]
            self.remove(p)

        for j in range(right_idx, l):
            p = points[j]
            self.remove(p)
        pass

    def isIn(self, projection):
        y1, x1, y2, x2 = self.bbox
        x, y = projection.data
        if x >= x1 and x <= x2 and \
                y >= y1 and y <= y2:
            return True
        return False
        pass

    def merge(self, other):
        assert self.label == other.label
        for k, p in other.points.items():
            self.points[k] = p
        return self

    def findRoI(self, frame_key):
        for record in self.records:
            if record[1] == frame_key:
                if record[0] is None:
                    return self
                return record[0]
        raise Exception("Not Found!")
        pass

    def check_box_sim(self, roi):
        y1, x1, y2, x2 = self.bbox
        h1 = y2 - y1
        w1 = x2 - x1

        alpha1 = h1 / float(w1)

        y1, x1, y2, x2 = roi.bbox
        h2 = y2 - y1
        w2 = x2 - x1

        alpha2 = h2 / float(w2)

        # print("%s |alpha1 - alpha2|: %f" % (self.label, np.abs(alpha1 - alpha2)))
        if np.abs(alpha1 - alpha2) > 0.08:
            return False
        else:
            return True

    def map_keypoint_to_box(self, keypoint):
        y1, x1, y2, x2 = self.bbox
        shp = self.frame.img_grey().shape
        h = y2 - y1
        w = x2 - x1

        x = (keypoint[0] - x1) * shp[1] / w
        y = (keypoint[1] - y1) * shp[0] / h
        return (x, y)

    def map_keypoint_from_box(self, keypoint):
        y1, x1, y2, x2 = self.bbox
        shp = self.frame.img_grey().shape
        h = y2 - y1
        w = x2 - x1

        x = keypoint[0] * w / shp[1] + x1
        y = keypoint[1] * h / shp[0] + y1
        return (x, y)

    # associate with pixels
    def associate_with(self, mappoint, frame, pos):
        key = (mappoint.key, frame.seq)
        if self.observations.get(key, None) is None:
            self.observations[key] = pos
        else:
            # @todo : TODO check
            last_pos = self.observations[key]
            if last_pos != pos:
                raise Exception(
                    "The mappoint %s projectd position %d has bee associated with frame %s in a different position: %d" % (
                        mappoint, pos, frame, last_pos))
        return self

    def set_FromFrame(self, frame):
        self.frame = frame
        return self

    # @todo : TODO already matched
    def is_active(self):
        return True

    # @todo : TODO observed by camera, i.e, camera is able to capure 3d points of the landmark
    # we use projection test upon estimated 3d structures to see whether this is viewable by
    # the camera
    def viewable(self):
        return True

    def last_frame(self):
        last_observation = self.records[-1][0]
        if last_observation != None:
            return last_observation.frame
        else:
            return self.frame

    def predict(self, cur_frame=None):
        y1, x1, y2, x2 = self.projected_pos
        w1, h1 = x2 - x1, y2 - y1
        box = np.array([x1, y1, w1, h1])
        box0 = self.kf.predict(box)
        assert (len(box0.shape) == 1)
        self.predicted_states = np.array([box0[0], box0[1], box0[0] + box0[2], box0[1] + box0[3]])

        if cur_frame is not None:
            # use the velocity estimated for the last frame (containing the object)
            last_frame = self.last_frame()
            box1 = last_frame.predictors['OpticalFlow'].predict(box, cur_frame.img_grey())
            self.predicted_states = np.array([box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]])

        if DEBUG:
            logging.info("<Landmark %d> , bbox <%d, %d, %d, %d>; predicted bbox by KalmanFilter: <%d, %d, %d, %d>" %
                         (self.seq, x1, y1, x2, y2,
                          self.predicted_states[0], self.predicted_states[1],
                          self.predicted_states[2], self.predicted_states[3]))

            logging.info("<Landmark %d> , bbox velocity (delta/frame) <%d, %d, %d, %d>, predicted by KalmanFilter" %
                         (self.seq,
                          self.predicted_states[0] - x1,
                          self.predicted_states[1] - y1,
                          box0[2] - w1,
                          box0[3] - h1))

            if cur_frame is not None:
                logging.info(
                    "<landmark %d> , bbox velocity (delta/frame) <%d, %d, %d, %d>, predicted by OpticalFlowBBoxPredictor" %
                    (self.seq,
                     box1[0] - x1,
                     box1[1] - y1,
                     box1[2] - w1,
                     box1[3] - h1))

        return self.predicted_states

    def update(self, detection):
        logging.info("<Landmark %d> update projected pose from %s => %s" %
                     (self.seq, str(self.projected_pos), str(detection.projected_pos)))
        self.projected_pos = detection.projected_pos
        self.projected_mask = detection.projected_mask
        #
        y1, x1, y2, x2 = self.projected_pos
        box = np.array([x1, y1, x2 - x1, y2 - y1])
        self.kf.update(box)

    ## Compute Cuboids
    def computeAABB(self):
        points = []
        for _, point in self.points.items():
            points.append(point.data)

        l = len(points)
        if l < 10:
            raise Exception("Not Enough Points (%d) for this landmark %s!" % (l, self))

        points = np.array(points)

        bottomLeftVec = points.min(axis=0)
        topRightVec = points.max(axis=0)

        bottomLeft = Point3D(bottomLeftVec[0], bottomLeftVec[1], bottomLeftVec[2])
        topRight = Point3D(topRightVec[0], topRightVec[1], topRightVec[2])

        # update AABB
        self._aabb = AABB(bottomLeft, topRight)
        return self._aabb

    # compute major direction and minor direction
    def computeOBB(self):
        if self._aabb is None:
            self.computeAABB()

        if self._obb is None:
            # estimate using PCA algortithm

            # @todo : TODO

            pass
        else:
            # do refinement

            #
            return self._obb

    def Centroid(self):
        v = np.zeros(3)
        l = len(self.points)
        for point_key, point in self.points.items():
            v += point.data
        v /= float(l)
        if self._centroid is None:
            self._centroid = Point3D(v[0], v[1], v[2])
        else:
            self._centroid.update(v[0], v[1], v[2])
        return self._centroid

    ## Spanning Tree

    def findParent(self):
        if self.parent is not None:
            parent = self.parent.findParent()
            self.parent = parent
            return parent
        return self

    def add(self, mappoint):
        if self.points.get(mappoint.key, None) is None:
            self.points[mappoint.key] = mappoint
        else:
            raise Exception("The point has already been registered into the RoI")

    def remove(self, mappoint):
        if self.points.get(mappoint.key, None) is not None:
            return self.points.pop(mappoint.key)
        else:
            logging.warning("The mappoint %s with key %s is not registered with the landmark!" % (
                mappoint, mappoint.key
            ))

    ## Just for logging

    def __str__(self):
        return "<F#%d.Ob#%d(%s:%.2f)>" % (self.frame.seq, self.seq, self.label, self.score)

    def __repr__(self):
        return "<F#%d.Ob#%d(%s)>" % (self.frame.seq, self.seq, self.label)