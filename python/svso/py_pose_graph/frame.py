import numpy as np
from enum import Enum
import cv2

# we will use linear_assignment to quickly write experiments,
# later a customerized KM algorithms with various optimization in c++ is employed
# see https://github.com/berhane/LAP-solvers

# This is used for "Complete Matching" and we can remove unreasonable "workers" first and then apply it
import scipy.optimize as Optimizer

# This is used for "Maximum Matching". There is a desired algorithm implementation for our references
import scipy.sparse.csgraph as Graph

import logging
from svso.lib.log import LoggerAdaptor

_logger = logging.getLogger("frame")

import numpy as np
import threading

from svso.py_pose_graph.point3d import Point3D
from svso.predictors import OpticalFlowBBoxPredictor, OpticalFlowKPntPredictor

try:
    from svso.lib.misc import AtomicCounter
    # AtomicCounter
except:
    class AtomicCounter(object):

        def __init__(self):
            self._counter = 0
            self.lock = threading.Lock()

        def incr(self):
            with self.lock:
                self._counter += 1
                return self._counter

        def __call__(self):
            return self.incr()

try:
    from svso.lib.misc import Identity
    # Identity
except:
    class Identity:
        def __init__(self, seq, name=None, id=None, uuid=None, tok=None, key=None):
            self.seq = seq
            self.name = name
            self.id = id
            self.uuid = uuid
            self.tok = tok
            self.key = key

        def __str__(self):
            return "Identity#%d" % self.seq

from svso.optimizers.bundle_adjust import g2o
from svso.lib.visualizer import display

R_PRECISION = 1e-4

class Pixel2D:

    Seq = AtomicCounter()

    # implements STL iterators
    class VectorIterator(object):
        def __init__(self, vec):
            self._vec = vec
            self.counter = self.__counter__()

        def __iter__(self):
            return self

        def __counter__(self):
            l = len(self._vec)
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
                return self._vec[ind]
            except StopIteration:
                raise StopIteration()

        def __str__(self):
            return "Vector iterator"

    def __init__(self, r, c, val=0):
        self.identity = Identity(Pixel2D.Seq())

        # y
        self.r = r

        # x
        self.c = c

        self.val = val

        # cv keypoint reference
        self.kp = None

        # key point extracted feature
        self.feature = None

        ## Covisibility Graph Topology

        # reproj 3d points in camera space
        #
        self.parent = None
        self.distance = None

        self.cam_pt_array = []

        # a weak frame reference
        self.frame = None

        # a weak roi reference
        self.roi = None

    @property
    def x(self):
        return self.c

    @property
    def y(self):
        return self.r

    @property
    def sources(self):
        return self.cam_pt_array

    def __getitem__(self, i):
        data = np.array([self.c, self.r])
        return data[i]

    def __setitem__(self, k, v):
        data = np.array([self.c, self.r])
        data[k] = v
        self.r = data[1]
        self.c = data[0]
        return self

    def set_frame(self, frame):
        self.frame = frame
        idx = self.pixel_pos
        # add strong connection
        frame.pixels[idx] = self
        return self

    @property
    def seq(self):
        return self.identity.seq

    @property
    def pixel_pos(self):
        frame = self.frame
        H, W = frame.img.shape[:2]
        idx = int(self.r * W + self.c)
        self.identity.key = idx
        return idx

    def set_roi(self, landmark):
        self.roi = landmark
        return self

    def set_feature(self, feature):
        self.feature = feature
        return self

    def set_kp(self, kp):
        self.kp = kp
        return self

    def add(self, cam_pt3):
        # if not present
        self.cam_pt_array.append(cam_pt3)
        return self

    def __iter__(self):
        data = np.array([self.c, self.r])
        return self.VectorIterator(data)

    @property
    def data(self):
        ret_data = np.array([int(self.c), int(self.r)])
        return ret_data

    def findObservationOf(self, world_pt3):
        if len(self.sources) == 0:
            return None
        if world_pt3 is None:
            return None

        best = None

        def check_eq(retrived, new):
            left = retrived.data
            right = new.data
            dist = np.linalg.norm(left - right)
            if dist < R_PRECISION:
                return True
            else:
                return False

        for p in self.cam_pt_array:
            # comparing world point location
            w = p.world
            if w is None:
                continue
            if check_eq(w, world_pt3):
                if best is None:
                    best = w
                else:
                    dist1 = np.linalg.norm(w - world_pt3)
                    dist2 = np.linalg.norm(best - world_pt3)
                    if dist1 < dist2:
                        best = w

        return best

    def isObservationOf(self, mappoint):
        if mappoint is None:
            return False
        frame = self.frame
        px_pos = mappoint.frames.get(frame.seq, None)
        if px_pos is None:
            return False
        else:
            idx = self.pixel_pos
            if idx == px_pos:
                return True
            else:
                return False

    # Union find with compression
    def findParent(self, cmpr=False):
        if self.parent is not None:
            parent = self.parent.findParent()
            if cmpr:
                self.parent = parent
            return parent
        return self

    def __repr__(self):
        return "<Pixel2D %d: r=%.3f, c=%.3f>" % (self.identity.seq, self.r, self.c)

    def __str__(self):
        return "<Pixel2D %d: r=%.3f, c=%.3f>" % (self.identity.seq, self.r, self.c)


from svso.models.sfe import SemanticFeatureExtractor


class Frame:

    Seq = AtomicCounter()
    logger = LoggerAdaptor("Frame", _logger)

    def __init__(self):
        self.identity = Identity(Frame.Seq())

        self.isKeyFrame = False

        ## Content

        # used to align with ground truth
        self.timestamp = None

        # might be a image path or url read by a asynchronous reader
        self._img_src = None

        # color constructed from img: cv::Mat or std::vector<byte>
        self.img = None

        # computed grey img or collected grey img directly from a camera
        self._img_grey = None

        # a camera instance to performance MVP or other image related computation
        self.camera = None

        # group of map tiles, where we storage 3d points, measurements and source images
        # Note: this should be a weak reference to the original data representation
        self.runtimeBlock = None

        ## Rigid object movements
        # later we will move these attributes to Object3D as common practice
        # in game development area, i.e, class Frame -> class Frame: public Object3D

        # rotation and translation relative to origins
        # Rwc. Note the author of ORB_SLAM2 uses the opporsite notation (Rcw) to reprented camera
        # camera pose, see discuss #226
        self.R0 = np.eye(3)
        # twc
        self.t0 = np.zeros((3, 1))

        # rotation and translation relative to the last frame, updated in each frame
        self.R1 = np.eye(3)
        self.t1 = np.zeros((3, 1))

        # used to very reprojection error in initialization process
        self.Rcw = np.eye(3)  # inv(R0)
        self.tcw = np.zeros((3, 1))  # -inv(R0).dot(t0)

        # GT set by tracker when timestamp and ground truth are all available
        self.assoc_timestamp = None
        self.R_gt = None
        self.t_gt = None

        ## Covisibility Graph Topology
        # previous frame
        self.pre = None

        #
        self.pixels = {}

        ## Features Expression Layer

        # extracted features
        # opencv key points
        self.kps = None
        # opencv key point features
        self.kps_feats = None

        # extracted roi keypoints (defaults to ORB keypoints)
        self.roi_kps = None
        # extracted roi features
        self.roi_feats = None

        # meta data
        self._detections = {}

        # media scene depth
        self.medianSceneDepth = -1.

        self.extractors = {}

        self.is_First = False

        ### Main Logics Executor ###

        #
        self.predictors = {
            'OpticalFlow': OpticalFlowBBoxPredictor(),
            'OpticalFlowKPnt': OpticalFlowKPntPredictor()
        }

        #
        self.matchers = {}

        #
        self.USE_IMAGE_LEVEL_ORB = False  # =>

        #
        self.SHOW_ROI = False

        #
        self.sample_size = -1

    @property
    def seq(self):
        return self.identity.seq

    @seq.setter
    def seq(self, val):
        self.identity.seq = val
        return self

    def set_camera(self, camera):
        self.camera = camera
        return self

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp
        return self

    # @todo : TODO
    def set_FromImg(self, img):
        self.img = img

        self.predictors['OpticalFlow'].set_FromImg(self.img_grey()).Init()
        self.predictors['OpticalFlowKPnt'].set_FromImg(self.img_grey()).Init()
        return self

    # getter of self._grey_img
    def img_grey(self):
        img_grey = self._img_grey
        if img_grey is None:
            img_grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            # suppose the image is undistorted
            self._img_grey = img_grey
        return img_grey

    # @todo : TODO
    def extract(self):
        orb_kp, orb_desc = (None, None)
        if self.USE_IMAGE_LEVEL_ORB:
            # @todo : REFACTOR the tasks should be running in pararllel
            # self.logger.info("%s, Extracting Image ORB features ..." % self)
            print("%s, Extracting Image ORB features ..." % self)
            orb_kp, orb_desc = self.ExtractORB()

            # @todo : TODO
            # add to frame

            # self.logger.info("Type of image orb key points : %s, size %d" % (type(orb_kp), len(orb_kp)))
            print("Type of image orb key points : %s, size %d" % (type(orb_kp), len(orb_kp)))
            # self.logger.info("Type of image orb descriptors : %s, shape %s" % (type(orb_desc), orb_desc.shape))
            print("Type of image orb descriptors : %s, shape %s" % (type(orb_desc), orb_desc.shape))

        # extract deep features or ROI
        # self.logger.info("%s, Extracting ROI features ..." % self)
        print("%s, Extracting ROI features ..." % self)
        roi_kp, roi_features = self.ExtractROI()

        kps = []
        kps_feats = []

        if self.USE_IMAGE_LEVEL_ORB:
            kps.extend(orb_kp)
            kps_feats.extend(orb_desc)
        # catenate orb keypoints and features, see opencv docs for definition
        # of returned key points and descriptors
        if self.extractors['sfe'].USE_ROI_LEVEL_ORB:
            for i, roi_feat_per_box in enumerate(roi_features):
                desc_per_box, kps_per_box = roi_feat_per_box['roi_orb']
                if len(kps_per_box) is 0:
                    label = roi_feat_per_box['label']
                    print("extract 0 points for detection#%d(%s)." % (i, label))
                    # raise ValueError("'kps_per_box' should not be an empty list!")
                kps.extend(kps_per_box)
                kps_feats.extend(desc_per_box)

        self.kps = kps
        self.kps_feats = kps_feats
        self.roi_kp = roi_kp
        self.roi_features = roi_features

        return (kps, kps_feats, roi_kp, roi_features)

    def ExtractORB(self, bbox=None, mask=None, label=None):
        # using opencv ORB extractor
        orb = self.extractors.get('orb', None)
        if orb is None:
            orb = cv2.ORB_create(edgeThreshold=5,
                                 patchSize=31,
                                 nlevels=8,
                                 fastThreshold=20,
                                 scaleFactor=1.2,
                                 WTA_K=4,
                                 scoreType=cv2.ORB_HARRIS_SCORE,
                                 firstLevel=0,
                                 nfeatures=1000)
            self.extractors['orb'] = orb

        img_grey = self.img_grey()
        shp = img_grey.shape
        if bbox is not None:
            y1, x1, y2, x2 = bbox

            ## adjust parameters
            h = y2 - y1
            w = x2 - x1

            # print("label %s, h:%d, w:%d" % (label, h, w))

            if np.min([w, h]) < 50:
                orb.setEdgeThreshold(0)
                orb.setPatchSize(15)
                orb.setScaleFactor(1.4)  # 1.4**8 ~ 8
                orb.setWTA_K(2)
            else:
                # restore
                orb.setEdgeThreshold(5)
                orb.setPatchSize(31)
                orb.setScaleFactor(1.2)
                orb.setWTA_K(4)
            ##

            # crop image

            new_img_grey = np.zeros(shp)
            new_img_grey[y1:y2, x1:x2] = img_grey[y1:y2, x1:x2]
            if self.SHOW_ROI:
                display(new_img_grey)
            img_grey = img_grey[y1:y2, x1:x2]
            img_grey = cv2.resize(img_grey, (shp[0], shp[1]), interpolation=cv2.INTER_CUBIC)
            # img_grey = cv2.cvtColor(new_img_grey.astype('uint8'), cv2.COLOR_GRAY2BGR)

        # compute key points vector
        kp = orb.detect(img_grey, None)

        # compute the descriptors with ORB
        kp, des = orb.compute(img_grey, kp)

        if bbox is not None:
            y1, x1, y2, x2 = bbox
            h = y2 - y1
            w = x2 - x1
            shp0 = img_grey.shape

            def _mapping(keypoint):
                x = keypoint.pt[0] * w / shp0[1] + x1
                y = keypoint.pt[1] * h / shp0[0] + y1
                keypoint.pt = (x, y)
                return keypoint

            kp = list(map(lambda p: _mapping(p),
                          kp))
            # kp = list(map(lambda idx: cv2.KeyPoint(kp[idx].x + x1, kp[idx].y + y1), indice))
        if bbox is not None and len(kp) > self.sample_size and self.sample_size is not -1:
            indice = np.random.choice(len(kp), self.sample_size)

            kp = list(map(lambda idx: kp[idx], indice))
            des = list(map(lambda idx: des[idx], indice))

        # filter out kp, des with mask
        # @todo : TODO

        # assert(len(kp) > 0)
        if len(kp) == 0:
            return [], []

        return kp, des

    def ExtractROI(self):
        # using our semantic features extractor
        sfe = self.extractors.get('sfe', None)
        if sfe is None:
            sfe = SemanticFeatureExtractor()
            self.extractors['sfe'] = sfe
            sfe.attach_to(self)

        # defaults to opencv channel last format
        img = self.img

        detections = sfe.detect(img)
        self._detections = detections

        # compute the descriptors with our SemanticFeaturesExtractor.encodeDeepFeatures
        kp, des = sfe.compute(img, detections)
        return kp, des

    def mark_as_first(self):
        self.is_First = True
        return self

    def find_px(self, x, y):
        H, W = self.img.shape[0:2]
        idx = int(y * W + x)
        return self.pixels.get(idx, None)

    def copy_pose_from(self, other_frame):
        self.R0 = other_frame.R0
        self.t0 = other_frame.t0
        camera = self.camera
        if camera is None:
            camera = Camera.clone(other_frame.camera)
            self.camera = camera
        return self

    def update_pose(self, pose):
        # see g2opy/python/types/slam3d/se3quat.h for details for the interface
        # print("R before update shape: %s, data: \n%s\n" % (self.R0.shape, self.R0))
        self.R0 = pose.orientation().matrix()
        # print("R after update shape: %s, data: \n%s\n" % (self.R0.shape, self.R0))

        # print("t before update shape: %s, data: \n%s\n" % (self.t0.shape, self.t0))
        self.t0 = pose.position().reshape((3, 1))
        # print("t after update shape: %s, data: \n%s\n" % (self.t0.shape, self.t0))

        # update R1, t1
        if self.pre is not None:
            self.R1 = self.R0.dot(np.linalg.inv(self.pre.R0))
            self.t1 = self.t0 - self.R1.dot(self.pre.t0)

            # update camera
            self.camera.R0 = self.R0
            self.camera.t0 = self.t0
            self.camera.R1 = self.R1
            self.camera.t1 = self.t1
        else:
            pass
        return self

    def get_pose_mat(self):
        if False:  # self.camera is not None:
            # print("camera.t0 shape: %s, data: \n%s\n" % (self.camera.t0.shape, self.camera.t0))
            # print("frame.t0  shape: %s, data: \n%s\n" % (self.t0.shape, self.t0))
            pose = g2o.SE3Quat(self.camera.R0, self.camera.t0.reshape(3, ))
        else:
            pose = g2o.SE3Quat(self.R0, self.t0.reshape(3, ))
        return pose.matrix()

    # retrieved current filled structure
    def get_points(self):
        pixels, cam_pts, points = [], [], []

        # @todo : TODO return current frame 3d structure
        for _, pixel in self.pixels.items():
            for cam_pt in pixel.sources:
                world = cam_pt.world
                if world is None:
                    continue
                points.append(world)

        return points

    def get_measurements(self):
        pixels, cam_pts, points = [], [], []

        # @todo : TODO return current frame 3d structure
        for _, pixel in self.pixels.items():
            for cam_pt in pixel.sources:
                world = cam_pt.world
                if world is None:
                    continue
                points.append(world)
                pixels.append(pixel)

        return points, pixels

    def get_landmarks(self):
        landmarks = set()
        for _, pixel in self.pixels.items():
            if pixel.roi is not None:
                landmarks.add(pixel.roi.findParent())

        return list(landmarks)

    # def match(self, reference_frame, kps, kps_feat):
    #     frame_key = reference_frame.seq
    #     img_shp = self.img.shape[0:2]
    #
    #     H, W = self.img.shape[0:2]
    #     matches = []
    #     unmatched_detections = []
    #     THR = 0.7  # 0.95 #0.8 #0.7
    #
    #     def _hamming_distance(x, y):
    #         from scipy.spatial import distance
    #         return distance.hamming(x, y)
    #
    #     def _get_neighbors(R, row, col, feat_map, img_shp):
    #         H, W = img_shp[0:2]
    #         x1, y1 = (col - R, row - R)
    #         x2, y2 = (col + R, row + R)
    #
    #         if x1 < 0:
    #             x1 = 0
    #         if y1 < 0:
    #             y1 = 0
    #
    #         if x2 >= W:
    #             x2 = W - 1
    #         if y2 >= H:
    #             y2 = H - 1
    #
    #         indice = feat_map[y1:y2, x1:x2] != -1
    #         return feat_map[y1:y2, x1:x2][indice]
    #
    #     #
    #     inv_kp = {}
    #     for i, kp in enumerate(kps):
    #         x, y = kp.pt
    #         idx = int(y * W + x)
    #         inv_kp[idx] = i
    #
    #     # init feat_map
    #     feat_map = np.full(img_shp, -1)
    #     for i, kp in enumerate(kps):
    #         x, y = kp.pt
    #         if int(y) >= img_shp[0] or int(y) < 0 or \
    #                 int(x) >= img_shp[1] or int(x) < 0:
    #             continue
    #         feat_map[int(y), int(x)] = i
    #
    #     cur_kp, cur_kp_features = self.kps, self.kps_feats
    #
    #     lost_pixels = 0
    #     skipped_in_parent_searching = 0
    #     skipped_in_neighbors_searching = 0
    #     dist_larger_than_thr = 0
    #
    #     print("reference frame #%d key points: %d" % (reference_frame.seq, len(kps)))
    #     print("cur frame #%d key points: %d" % (self.seq, len(cur_kp)))
    #     if frame_key is not self.seq:
    #         for i, kp in enumerate(cur_kp):
    #             x, y = kp.pt
    #             idx = int(y * W + x)
    #             px = self.pixels.get(idx, None)
    #             if px is None:
    #                 # print("The pixel (%f,%f) is none!" % (x, y))
    #                 lost_pixels += 1
    #                 continue
    #
    #             parent = px.parent
    #
    #             if parent is None:
    #                 # print("The parent of the pixel is none!")
    #                 unmatched_detections.append(px)
    #                 continue
    #
    #             feat_l = cur_kp_features[i]
    #
    #             # print("=====================")
    #             # print("searching starts from %s(Frame#%d) 's parent %s(Frame#%d)" % (px, px.frame.seq, parent, parent.frame.seq))
    #             flag = True
    #             while parent is not None and flag:
    #                 if parent.frame.seq == frame_key:
    #                     # print("find a matched pixel %s(Frame#%d) for Frame#%d" % (parent, parent.frame.seq, frame_key))
    #                     flag = False
    #                 else:
    #                     parent = parent.parent
    #                     if parent is not None:
    #                         # print("trace back to pixel %s(Frame#%d)" % (parent, parent.frame.seq))
    #                         pass
    #
    #             if flag:
    #                 skipped_in_parent_searching += 1
    #                 continue
    #
    #             # print("\n")
    #             # print("find a match for pixel %s" % px)
    #             # find a match
    #             #         jdx = int(parent.r * W + parent.c)
    #             #         j = inv_kp[jdx]
    #
    #             #         feat_r = kps_feat[j]
    #             #         feat_r = parent.feature
    #             #         dist = _hamming_distance(feat_l, feat_r)
    #
    #             # perform local search
    #             indice = _get_neighbors(8, int(parent.r), int(parent.c), feat_map, img_shp)
    #             if len(indice) == 0:
    #                 skipped_in_neighbors_searching += 1
    #                 continue
    #
    #             # KNN search
    #             feat_l = cur_kp_features[i]
    #
    #             dist = None
    #             min_dist, min_ind = np.inf, None
    #
    #             for ind in indice:
    #                 feat_r = kps_feat[ind]
    #                 dist = _hamming_distance(feat_l, feat_r)
    #                 if min_dist > dist:
    #                     min_dist = dist
    #                     min_ind = ind
    #
    #             if min_dist >= THR:
    #                 dist_larger_than_thr += 1
    #                 continue
    #
    #             # matches.append(cv2.DMatch(i, j, dist))
    #             matches.append(cv2.DMatch(i, min_ind, min_dist))
    #
    #     else:
    #         # see Tracker._match
    #         raise Exception("Not Implemented Yet")
    #
    #     matches = sorted(matches, key=lambda mtch: mtch.distance)
    #     import pandas as pd
    #
    #     print("lost pixels : %d" % lost_pixels)
    #     print("unmatched detections : %d" % len(unmatched_detections))
    #     print("skipped when searching parent : %d" % skipped_in_parent_searching)
    #     print("skipped when searching neighbors of parents (8PX neighbors) : %d" % skipped_in_neighbors_searching)
    #     print("skipped when min dist is larger than %f : %d" % (THR, dist_larger_than_thr))
    #     if len(matches) < 10:
    #         raise Exception("Unwanted matching results!")
    #
    #     distances = [mtch.distance for mtch in matches]
    #
    #     df = pd.DataFrame({
    #         "Dist": distances
    #     })
    #
    #     print(df)
    #
    #     l = len(self.kps)
    #     mask = np.ones((l, 1))
    #
    #     print("Found %d matches using Union Find algorithm" % len(matches))
    #
    #     return matches, mask.tolist()

    def match(self, reference_frame, kps, kps_feat):
        frame_key = reference_frame.seq
        img_shp = self.img.shape[0:2]

        H, W = self.img.shape[0:2]
        matches = []
        unmatched_detections = []
        THR = 0.75
        THR2 = 0.75

        def _hamming_distance(x, y):
            from scipy.spatial import distance
            return distance.hamming(x, y)

        def _get_neighbors(R, row, col, feat_map, img_shp):
            H, W = img_shp[0:2]
            x1, y1 = (col - R, row - R)
            x2, y2 = (col + R, row + R)

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            # freshly added (!important)
            if x2 < 0:
                x2 = 0
            if y2 < 0:
                x2 = 0

            if x2 >= W:
                x2 = W - 1
            if y2 >= H:
                y2 = H - 1

            indice = feat_map[y1:y2, x1:x2] != -1
            return feat_map[y1:y2, x1:x2][indice]

        #
        inv_kp = {}
        for i, kp in enumerate(kps):
            x, y = kp.pt
            idx = int(y * W + x)
            inv_kp[idx] = i

        # init feat_map
        feat_map = np.full(img_shp, -1)
        for i, kp in enumerate(kps):
            x, y = kp.pt
            if int(y) >= img_shp[0] or int(y) < 0 or \
                    int(x) >= img_shp[1] or int(x) < 0:
                continue
            feat_map[int(y), int(x)] = i

        cur_kp, cur_kp_features = self.kps, self.kps_feats

        lost_pixels = 0
        skipped_in_parent_searching = 0
        skipped_in_neighbors_searching = 0
        dist_larger_than_thr = 0

        print("reference frame #%d key points: %d" % (reference_frame.seq, len(kps)))
        print("cur frame #%d key points: %d" % (self.seq, len(cur_kp)))
        dx, dy = 0, 0
        cnt = 0
        if frame_key is not self.seq:
            for i, kp in enumerate(cur_kp):
                x, y = kp.pt
                idx = int(y * W + x)
                px = self.pixels.get(idx, None)
                if px is None:
                    # print("The pixel (%f,%f) is none!" % (x, y))
                    lost_pixels += 1
                    continue

                parent = px.parent

                if parent is None:  # deactivate the following searching method
                    # print("The parent of the pixel is none!")
                    unmatched_detections.append((i, kp, px))
                    continue

                feat_l = cur_kp_features[i]

                # print("=====================")
                # print("searching starts from %s(Frame#%d) 's parent %s(Frame#%d)" % (px, px.frame.seq, parent, parent.frame.seq))
                flag = True
                while parent is not None and flag:
                    if parent.frame.seq == frame_key:
                        # print("find a matched pixel %s(Frame#%d) for Frame#%d" % (parent, parent.frame.seq, frame_key))
                        flag = False
                    else:
                        parent = parent.parent
                        if parent is not None:
                            # print("trace back to pixel %s(Frame#%d)" % (parent, parent.frame.seq))
                            pass

                if flag:
                    skipped_in_parent_searching += 1
                    unmatched_detections.append((i, kp, px))
                    continue

                # print("\n")
                # print("find a match for pixel %s" % px)
                # find a match
                #         jdx = int(parent.r * W + parent.c)
                #         j = inv_kp[jdx]

                #         feat_r = kps_feat[j]
                #         feat_r = parent.feature
                #         dist = _hamming_distance(feat_l, feat_r)

                dx += parent.c - px.c
                dy += parent.r - px.r
                cnt += 1

                # perform local search
                indice = _get_neighbors(10, int(parent.r), int(parent.c), feat_map, img_shp)
                if len(indice) == 0:
                    skipped_in_neighbors_searching += 1
                    continue

                # KNN search
                feat_l = cur_kp_features[i]

                dist = None
                min_dist, min_ind = np.inf, None

                for ind in indice:
                    feat_r = kps_feat[ind]
                    dist = _hamming_distance(feat_l, feat_r)
                    if min_dist > dist:
                        min_dist = dist
                        min_ind = ind

                if min_dist >= THR:
                    dist_larger_than_thr += 1
                    unmatched_detections.append((i, kp, px))
                    continue

                # matches.append(cv2.DMatch(i, j, dist))
                matches.append(cv2.DMatch(i, min_ind, min_dist))

        else:
            # see Tracker._match
            raise Exception("Not Implemented Yet")

        if cnt != 0:
            dx /= cnt
            dy /= cnt
            print("Average dx:", dx)
            print("Average dy:", dy)

        #     optical_flow = OpticalFlowKPntPredictor()
        #     optical_flow.Init()
        #     optical_flow.set_FromImg(self.img_grey())

        #     # compute accumulative flows
        #     pre = self.pre
        #     flow = pre.predictors['OpticalFlowKPnt'].get_flow()
        #     assert flow.shape[:2] == img_shp
        #     while pre.seq is not frame_key:
        #       pre = pre.pre
        #       flow1 = pre.predictors['OpticalFlowKPnt'].get_flow()
        #       assert flow.shape == flow1.shape
        #       flow += flow1

        #     # predict coord
        #     kps_coor_r = list(map(lambda kp: kp.pt, kps))

        #     # init feat_map
        #     feat_map = np.full(img_shp, -1)
        #     for i, kp in enumerate(kps_coor_r):
        #       x, y = kp
        #       shift_x = flow[int(y), int(x), 0]
        #       shift_y = flow[int(y), int(x), 1]
        #       x += shift_x
        #       y += shift_y
        #       if int(y) >= img_shp[0] or int(y) < 0 or \
        #          int(x) >= img_shp[1] or int(x) < 0:
        #         continue
        #       feat_map[int(y),int(x)] = i

        for i, kp, px in unmatched_detections:
            #       if cnt > 10:
            #         target = [px.x+dx, px.y+dy] # slightly bad
            #       else:
            #         target = optical_flow.predict([px.data], reference_frame.img_grey())[0] # not stable

            s = px.roi.map_keypoint_to_box(px.data)
            try:
                ref_roi = px.roi.findParent().findRoI(frame_key)  # very good
            except Exception as e:
                # print(e)
                continue
            if ref_roi.check_box_sim(px.roi):
                target = ref_roi.map_keypoint_from_box(s)
                # update dx , dy
                dx = (dx * cnt + target[0] - px.c) / (cnt + 1)
                dy = (dy * cnt + target[1] - px.r) / (cnt + 1)
                cnt += 1
            else:
                #         target = [px.x+dx, px.y+dy]
                continue

            #       target = [px.x, px.y] # very bad
            indice = _get_neighbors(12, int(target[1]), int(target[0]), feat_map, img_shp)

            #       indice = _get_neighbors(7, int(px.y), int(px.x), feat_map, img_shp) # bad and not stable

            if len(indice) == 0:
                continue

            # KNN search
            feat_l = cur_kp_features[i]

            dist = None
            min_dist, min_in = np.inf, None

            for ind in indice:
                feat_r = kps_feat[ind]
                dist = _hamming_distance(feat_l, feat_r)
                if min_dist > dist:
                    min_dist = dist
                    min_ind = ind

            if cnt > 10:

                if min_dist >= THR:
                    continue

            else:

                if min_dist >= THR2:
                    continue

            kp_in_ref = kps[min_ind]
            x, y = kp_in_ref.pt
            jdx = int(y * W + x)
            px_in_ref = reference_frame.pixels.get(jdx, None)
            if px_in_ref is not None and px.parent is None:
                px.parent = px_in_ref
            matches.append(cv2.DMatch(i, min_ind, min_dist))

        matches = sorted(matches, key=lambda mtch: mtch.distance)
        import pandas as pd

        print("lost pixels : %d" % lost_pixels)
        print("unmatched detections : %d" % len(unmatched_detections))
        print("skipped when searching parent : %d" % skipped_in_parent_searching)
        print("skipped when searching neighbors of parents (8PX neighbors) : %d" % skipped_in_neighbors_searching)
        print("skipped when min dist is larger than %f : %d" % (THR, dist_larger_than_thr))
        if len(matches) < 10:
            raise Exception("Unwanted matching results!")

        distances = [mtch.distance for mtch in matches]

        df = pd.DataFrame({
            "Dist": distances
        })

        print(df)

        l = len(self.kps)
        mask = np.ones((l, 1))

        print("Found %d matches using Union Find algorithm" % len(matches))

        return matches, mask.tolist()

    def __repr__(self):
        return "<Frame %d>" % self.identity.seq

    def __str__(self):
        return "<Frame %d>" % self.identity.seq


# use to compute perspective camera MVP projections with intrinsic parameters and distortion recover (using OpenCV4)
# the reading loop is implemented using CV2 camera.
class Camera:
    from enum import Enum
    class Status(Enum):
        MONOCULAR = 1

    class ProjectionType(Enum):
        PERSPECTIVE = 1
        UNSOPPROTED = -1

    def __init__(self, device, R, t, anchor_point, H=None, W=None):
        # default mode is monocular
        self.mode = Camera.Status.MONOCULAR
        self.type = Camera.ProjectionType.PERSPECTIVE

        #
        self.device = device

        # extrinsic parameters of a camera, see TUM vision group dataset format
        self.K = device.K

        # world
        self.R0 = None
        self.t0 = None

        # eye and pose
        self.R1 = R
        self.t1 = t

        self.Rcw = np.eye(3)
        self.tcw = np.zeros((3, 1))

        self.anchor_point = anchor_point

        self.H = H
        self.W = W

    def Init(self):
        self.K = self.device.K
        # update other computed properties
        return self

    @staticmethod
    def clone(camera):
        camera_cpy = Camera(camera.device, camera.R1, camera.t1, camera.anchor_point, H=camera.H, W=camera.W)
        camera_cpy.R0 = camera.R0
        camera_cpy.t0 = camera.t0
        return camera_cpy

    # @todo : TODO
    def t_SE3ToR3(self, t=None):
        if t is None:
            t = self.t1
        return np.array([
            [0., -t[2], t[1]],
            [t[2], 0., -t[0]],
            [-t[1], t[0], 0.]
        ])

    # @todo : TODO
    def viewWorldPoint(self, point3d):
        v = point3d.data.reshape(3, 1)

        cam_pt = np.linalg.inv(self.R0).dot(v - self.t0)

        ret = Point3D(cam_pt[0][0], cam_pt[1][0], cam_pt[2][0])
        # used for debug
        ret.seq = point3d.seq
        return ret

    # @todo : TODO
    def view(self, point3d):
        # should be homogenous point
        v = point3d.data.reshape(3, 1)
        if v[2][0] < 0:
            print("The point %s is not visible by the camerea" % point3d)
            return None

        # print("v(ori):\n",v)
        v /= v[2][0]
        # print("K:\n",self.K)
        # print("v:\n",v)
        px = self.K.dot(v)
        # print("px:\n", px)
        px = px.reshape((3,))
        px = Pixel2D(px[1], px[0])

        if px.x < 0 or px.x >= self.W or px.y < 0 or px.y >= self.H:
            print("The projection %s of the point %s is out of bound (%f, %f)" % (
                px,
                point3d,
                self.W,
                self.H
            ))
            return None
        return px

    def reproj(self, pixel2d):
        px = pixel2d
        K = self.K
        # compute normalized point in camera space
        if isinstance(px, cv2.KeyPoint):
            px = Pixel2D(px.pt[1], px.pt[0])

        if isinstance(px, tuple):
            px = Pixel2D(px[1], px[0])

        return Point3D(
            (px.x - K[0, 2]) / K[0, 0],
            (px.y - K[1, 2]) / K[1, 1],
            1
        )

