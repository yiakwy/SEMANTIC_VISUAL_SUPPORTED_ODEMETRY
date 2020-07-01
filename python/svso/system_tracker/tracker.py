import runtime_block_pb2 as runtime_block_proto

import os
import ctypes
import cv2
import numpy as np
from enum import Enum
import threading

import logging
from svso.lib.log import LoggerAdaptor

_logger = logging.getLogger("tracker")

from svso.lib.visualizer import WebImageRenderer
from svso.py_pose_graph.mapblock import RuntimeBlock
from svso.py_pose_graph.frame import Frame, Camera
from svso.py_pose_graph.roi import Observation
from svso.py_pose_graph.point3d import WorldPoint3D, CameraPoint3D
from svso.matcher import OpticalFlowBasedKeyPointsMatcher, ROIMatcher
from svso.optimizers.bundle_adjust import g2o, BundleAdjustment, PoseOptimization
from svso.localization.pnp import fetchByRoI, PnPSolver
from svso.lib.maths.rotation import Quaternion
from svso.lib.exceptions import RelocalizationErr
from svso.validation_toolkit.tum import TUM_DATA_DIR, query_depth_img, Program as Read_GT_Program
from svso.mapping.local_mapping import LocalMapping
from svso.mapping.relocalization import Relocalization
from svso.config import Settings

settings = Settings()

# setting debug variable
DEBUG = settings.DEBUG
USE_POSE_GROUND_TRUTH = settings.USE_POSE_GROUND_TRUTH

## == Legacy Codes Begin (use py_pose_graph/c_pose_graph instead!) ==

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


class Vec3(ctypes.Array):
    _type_ = ctypes.c_double
    _length_ = 3

    # # using python descriptor framework to mick behavior of anonymous union
    # def __getattr__(self, name):
    #     if name == "x":
    #         return self.__getitem__(0)
    #     if name == "y":
    #         return self.__getitem__(1)
    #     if name == "z":
    #         return self.__getitem__(2)

    @property
    def x(self):
        return self.__getitem__(0)

    @x.setter
    def x(self, val):
        self.__setitem__(0, val)
        return self

    @property
    def y(self):
        return self.__getitem__(1)

    @y.setter
    def y(self, val):
        self.__setitem__(1, val)
        return self

    @property
    def z(self):
        return self.__getitem__(2)

    @z.setter
    def z(self, val):
        self.__setitem__(2, val)
        return self

class Pose(ctypes.Union):
    _fields_ = ("azimuth", ctypes.c_double), ("pitch", ctypes.c_double), ("roll", ctypes.c_double), ("data", Vec3)

# helper classes
PIXEL_PRECISION = 3

class Point3D:
    # Simulate an atomic 64 bit integer as an index. Note python provides no
    # concepts like atomic operators (see c++) such that the compiled instructions
    # not be affected by disorder of threads execution sequence.
    Seq = AtomicCounter()

    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z  # depth value is it can be viewed by a camera

        # might not be selected fro triangulation
        self.triangulated = False

        #
        self._isBad = False

        self.type = "local"

        ## Covisibility Graph Topology

        # source
        self.world = None

        # If point is upgraded as keypoint, the attribute is used to trace back
        # union set, here i uses Uncompressed Union Set to present films of the object
        self.parent = None
        self.records = []

        # projected pixel
        self.px = None

        # set color
        self.color = None

        # used if the point is a world point
        self.frames = {}

        ## Identity

        # id
        self.id = None
        self.seq = self.Seq()
        # generate uuid using uuid algorithm
        self.uuid = ""

    def update(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        return self

    def set_color(self, color):
        self.color = color
        return self

    def setBadFlag(self):
        self._isBad = True
        return self

    def isBad(self):
        return self._isBad

    def __getitem__(self, i):
        # note i might be a tuple to be processed by a slicer, hence I use numpy for sanity checking
        data = np.array([self.x, self.y, self.z])
        return data[i]

    @property
    def data(self):
        data = np.array([self.x, self.y, self.z])
        return data

    def __setitem__(self, k, v):
        # note i might be a tuple to be processed by a slicer, hence I use numpy for sanity checking
        data = np.array([self.x, self.y, self.z])
        data[k] = v
        self.x = data[0]
        self.y = data[1]
        self.z = data[2]
        return self

    # @property
    # def data(self):
    #   _data = np.array([self.x, self.y, self.z])
    #   return _data

    def set_FromWorld(self, world):
        if not isinstance(world, self.__class__):
            raise ValueError("Expect world to be type %s but find %s\n" % (str(self.__class__), str(type(world))))
        self.world = world
        return self

    def set_FromType(self, type_str):
        self.type = type_str
        return self

    def associate_with(self, frame, pos):
        last_pos = self.frames.get(frame.seq, None)
        if last_pos is None:
            self.frames[frame.seq] = pos
            # check whether the pos is empty in frame
            if frame.pixels.get(pos, None) is None:
                raise Exception("The pixel is not recorded by the frame")
        else:
            if last_pos != pos:
                print("pos %d is different from last_pos: %d" % (pos, last_pos))
                print("Pixel(frame=%d, r=%d, c=%d, pos=%d) mapped points: " % \
                      (frame.seq, frame.pixels[pos].r, frame.pixels[pos].c, pos))
                for p in frame.pixels[pos].sources:
                    w = p.world
                    print(w)
                print("\n")
                print("Pixel(frame=%d, r=%d, c=%d, last_pos=%d) mapped points: " % \
                      (frame.seq, frame.pixels[last_pos].r, frame.pixels[last_pos].c, last_pos))
                for p in frame.pixels[last_pos].sources:
                    w = p.world
                    print(w)
                # if the two pixels are close enough
                if np.abs(frame.pixels[last_pos].r - frame.pixels[pos].r) < PIXEL_PRECISION and \
                        np.abs(frame.pixels[last_pos].c - frame.pixels[pos].c) < PIXEL_PRECISION:
                    pass
                raise Exception("The point %s has already been mapped to a different place" % self)
        return self

    # numpy wrapper. For fully scientific matrix container implementation see PyMatrix
    def __add__(self, other):
        if isinstance(other, [int, float]):
            return self.__class__(self.x + other, self.y + other, self.z + other)
        else:
            # check

            ret = self.data() + other.data()
            return self.__class__(ret[0], ret[1], ret[2])

    def __sub__(self, other):
        if isinstance(other, [int, float]):
            return self.__class__(self.x - other, self.y - other, self.z - other)
        else:
            # check

            ret = self.data() - other.data()
            return self.__class__(ret[0], ret[1], ret[2])

    @staticmethod
    def NotSupported():
        raise Exception("Not supported operation, please convert it to Vec3!")

    def __mul__(self, other):
        self.NotSupported()

    def __div__(self, other):
        self.NotSupported()

    def __str__(self):
        if self.type == "world":
            return "<Point3D %d: %.3f, %.3f, %.3f>" % (self.seq, self.x, self.y, self.z)
        else:
            return "<Camera Point3D %d: %.3f, %.3f, %.3f>" % (self.seq, self.x, self.y, self.z)

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
        self.r = r
        self.c = c
        self.val = val

        self.kp = None
        self.feature = None

        ## Covisibility Graph Topology

        # reproj 3d points in camera space
        #
        self.parent = None
        self.distance = None

        self.sources = []
        # a weak frame reference
        self.frame = None
        # a weak roi reference
        self.roi = None

        ## Identity

        # id
        self.id = None
        self.seq = self.Seq()
        # generate uuid using uuid algorithm
        self.uuid = None
        self.key = None

    def set_FromFrame(self, frame):
        self.frame = frame
        H, W = frame.img.shape[:2]
        idx = int(self.r * W + self.c)
        self.key = idx
        # add strong connection
        frame.pixels[idx] = self
        return self

    def set_FromROI(self, landmark):
        self.roi = landmark
        # H, W = self.frame.img.shape[:2]
        # add strong connecion
        # idx = int(self.r*W + self.c)
        # associate with frame
        # landmark.associate_with(self.frame, idx)
        return self

    def set_FromFeature(self, feature):
        self.feature = feature
        return self

    def set_FromKp(self, kp):
        self.kp = kp
        return self

    def add_camera_point(self, p3d):
        # if not present
        self.sources.append(p3d)
        return self

    def __getitem__(self, i):
        data = np.array([self.c, self.r])
        return data[i]

    @property
    def x(self):
        return self.c

    @property
    def y(self):
        return self.r

    def __setitem__(self, k, v):
        data = np.array([self.c, self.r])
        data[k] = v
        self.r = data[1]
        self.c = data[0]
        return self

    def __iter__(self):
        data = np.array([self.c, self.r])
        return self.VectorIterator(data)

    @property
    def data(self):
        _data = np.array([int(self.c), int(self.r)])
        return _data

    # @todo : TODO
    def findObservationOf(self, world_pt3):
        if len(self.sources) == 0:
            return None
        if world_pt3 is None:
            return None

        best = None
        frame = self.frame
        H, W = frame.img.shape[:2]
        idx = int(self.r * W + self.c)

        def check_eq(retrived, new):
            left = retrived.data()
            right = new.data()
            dist = np.linalg.norm(left - right)
            if dist < R_PRECISION:
                return True
            else:
                return False

        for p in self.sources:
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
            H, W = frame.img.shape[:2]
            idx = int(self.r * W + self.c)
            if idx == px_pos:
                return True
            else:
                return False

    # Union find with compression
    def findParent(self):
        if self.parent is not None:
            parent = self.parent.findParent()
            # self.parent = parent
            return parent
        return self

    def __repr__(self):
        return "<Pixel2D %d: r=%.3f, c=%.3f>" % (self.seq, self.r, self.c)

    def __str__(self):
        return "<Pixel2D %d: r=%.3f, c=%.3f>" % (self.seq, self.r, self.c)

## == Legacy Codes End ==

# class RelocalizationErr(Exception): pass

# This should be running in the main thread
class Tracker:
    class State(Enum):
        UNDEFINED_STATE = -1
        UNINITIALIZED = 1
        INITIALIZED = 2
        TRACKING = 3
        LOSTED = 4

    REF_STEP = 5

    def __init__(self):
        # main loop state
        self.state = Tracker.State.UNINITIALIZED

        # since there is no atomic operator around python, I have to warp the value with a lock manually
        self._initialized = False

        # used to decide whether to trigger more operation update the map
        self._is_relocalization_mode = False
        self._state_modifier_lock = threading.Lock()
        self._renderer = WebImageRenderer()
        self._map = None
        self.mapblocks = []
        self.mapblock_idx = 0

        self.matcher = None
        #     # previous key frames and cur frame used for triangulation
        #     self.key_frames = []
        # self.reference_frame = None
        self.cur = None

        self.ref = None

        #
        self.last_frame = None

        # frame rendering property
        self.flow_mask = None

        #
        self.cur_anchor_point = np.array([0., 0., 0.])

        # trajectory
        self.trajectory = [self.cur_anchor_point]

        # trajectory GT
        self.trajectories_gt = None # trajectories

        # depth images
        self.depth_images = []

        #
        self.SlidingWindow = []

        ## Specification of workers

        self.mapper = LocalMapping()
        self.mapper.set_FromTracker(self)
        # Not implemented yet
        self.loopCloser = None

        # Relocalizer
        self.relocalizer = Relocalization()
        self.relocalizer.set_FromTracker(self)

    def Init(self):
        pass

    def Reset(self):
        pass

    def Reset_Relocalization(self):
        # reset the tracker main loop state
        self.state = Tracker.State.UNINITIALIZED

        # hold map blocks and states relating to relocalization state

        # reset initialization states
        self._initialized = False

        # keep relocalization state

        self._state_modifier_lock = threading.Lock()
        self._renderer = WebImageRenderer()

        # keep references to mapblocks
        # _map and mapblock_idx has been changed outside of Reset_Relocalization function

        # reset algorithms modules to clear caches
        self.matcher = None

        # reset frames holders
        self.cur = None

        self.ref = None

        self.last_frame = None

        # reset rendering properties
        self.flow_mask = None

        #
        self.cur_anchor_point = np.array([0., 0., 0.])

        # keep trajectories

        # keep ground truth

        #
        self.SlidingWindow.clear()

        ## Reset workers

        # Reset Localizer
        if self.mapper is not None:
            self.mapper.set_stopping()

        self.mapper = LocalMapping()
        self.mapper.set_FromTracker(self)

        # Not implemented yet
        self.loopCloser = None

        # Reset Relocalizer
        if self.relocalizer is not None:
            self.relocalizer.set_stopping()

        self.relocalizer = Relocalization()
        self.relocalizer.set_FromTracker(self)
        return self
        pass

    def set_FromMap(self, map):
        if not isinstance(map, RuntimeBlock):
            raise ValueError("expect map of type <%s> but found %s" % (str(RuntimeBlock), str(type(map))))
        self._map = map
        self.mapper.set_FromMap(self._map)
        return self

    # @todo : TODO
    def _init(self):
        pass

    def _match(self, reference_frame):
        #### Step 1 : exract features

        cur_kp, cur_kp_features, cur_roi_kp, cur_roi_features = self.cur.extract()
        detected_objects = []
        for i, roi_feature in enumerate(cur_roi_features):
            detected_object = Observation(roi_feature['label'], cur_roi_kp[i], roi_feature).set_FromFrame(self.cur)
            detected_object.Init()
            detected_objects.append(detected_object)

            kp_per_box = cur_roi_kp[i]
            if len(kp_per_box) is 0:
                logging.warn("There is no detected key points(camera space) associated to %s" % detected_object)
            else:
                for px in kp_per_box:
                    x, y = px
                    for obj in detected_objects:
                        px.set_roi(obj)

        self.cur._detections = detected_objects
        #### Step 2 : track detected objects

        # track detected objects using landmarks with updated postions (bbox)
        rets = self._map.track(self.cur, detected_objects)
        if DEBUG and len(rets) > 0:
            mtched, unmtched_landmarks, unmtched_detections = rets

            rendered_img = self.cur.img.copy()
            draw_group = {}
            for mtch in mtched:
                fid = mtch[0].frame.seq
                frame = draw_group.get(fid, None)
                if frame is None:
                    draw_group[fid] = {
                        'frame': mtch[0].frame,
                        'matches': [],
                        'unmatches': []
                    }
                draw_group[fid]['matches'].append(mtch)

            for unmtch in unmtched_landmarks:
                fid = unmtch.frame.seq
                frame = draw_group.get(fid, None)
                if frame is None:
                    draw_group[fid] = {
                        'frame': unmtch.frame,
                        'matches': [],  # mtched detection and landmarks
                        'unmatches': []  # unmtched landmarks
                    }
                draw_group[fid]['unmatches'].append(unmtch)

            for fid, kw in draw_group.items():
                masked_img = self._renderer.drawMatchedROI(rendered_img,  # changed here
                                                           kw['frame'].img,
                                                           kw['matches'],
                                                           kw['unmatches'],
                                                           unmtched_detections)
                print("Plot matches between cur frame %s and frame %s" % (self.cur, kw['frame']))
                self._renderer.render(masked_img)
                rendered_img = self._renderer._masked_img

            self.cur.rendered_img = rendered_img

        if reference_frame is None:
            # see key frames selection principles discussion: https://github.com/raulmur/ORB_SLAM2/issues/872
            self._map.add_new_key_frame(self.cur)
            self.cur.isKeyFrame = True
            # setting current frame as reference
            self.ref = self.cur

            # using ground truth R, t
            # see key frames selection principles discussion: https://github.com/raulmur/ORB_SLAM2/issues/872
            self._map.add_new_key_frame(self.cur)
            self.cur.isKeyFrame = True
            # setting current frame as reference
            self.ref = self.cur

            # using ground truth R, t
            if USE_POSE_GROUND_TRUTH and self.cur.timestamp is not None:
                self.cur.R0 = self.cur.R_gt[0:3, 0:3]
                self.cur.t0 = self.cur.t_gt

                pass

            camera = Camera(self._map.device, self.cur.R1, self.cur.t1, self.cur.t0, H=self.cur.img.shape[0],
                            W=self.cur.img.shape[1])
            camera.R0 = self.cur.R0
            camera.t0 = self.cur.t0

            self.cur.camera = camera

            return False

        #### Step 3 : find pixel matches with the last frame (important!)
        # note in our new alogrithm, this is done every frame, see Tracker._match

        # compute matched key pionts
        mtches = []
        # matcher = FlannBasedKeyPointsMatcher().set_FromFrame(self.cur).set_FromFeatures(cur_kp_features)
        # matcher.Init()
        # self.cur.matchers['FlannBasedMatcher'] = matcher

        last_frame = self.get_last_frame()
        self.cur.pre = last_frame
        # kps_mtched, mask = matcher.mtch(last_frame.kps_feats)

        ## Freshly updated matcher
        matcher = OpticalFlowBasedKeyPointsMatcher() \
            .set_FromFrame(self.cur) \
            .set_FromKP(cur_kp) \
            .set_FromFeatures(cur_kp_features)

        self.cur.matchers['OpticalFlowBasedKeyPointsMatcher'] = matcher

        kps_mtched, mask = matcher.mtch(last_frame.kps, last_frame.kps_feats, last_frame)
        for i, mtched in enumerate(kps_mtched):
            (x1, y1) = cur_kp[mtched.queryIdx].pt
            (x2, y2) = last_frame.kps[mtched.trainIdx].pt

            cur_px = self.cur.find_px(x1, y1)
            last_px = last_frame.find_px(x2, y2)
            # (x3, y3) = last_px.findParent().data

            assert (cur_px != None)
            assert (last_px != None)

            # later we will use union findParent to fetch the px with largest disparity
            cur_px.parent = last_px

        ## add mtched keypoints to map

        logging.info("key points matched (kps_mtched) : %d" % len(kps_mtched))
        # do visualization of orb key points matching
        # draw matches from cur to frame

        self.last_frame = last_frame
        if DEBUG:
            flow_mask = self._renderer.drawOpticalFlow(last_frame)
            masked_img = cv2.add(last_frame.img, flow_mask)

            # for offline rendering
            self.flow_mask = flow_mask

        if len(kps_mtched) < 10:
            # logging.info("switch state from %s to %s" % (self.state, Tracker.State.LOSTED))
            # self.state = Tracker.State.LOSTED
            ###
            self.cur.copy_pose_from(self.get_last_frame())
            ###
            return

        #### Step 4 : triangluation with reference frame

        # retrieve matches with reference
        reference_frame = self.ref
        try:
            ref_kps_mtched, ref_mask = self.cur.match(reference_frame, reference_frame.kps, reference_frame.kps_feats)
        except Exception as e:
            print(e)
            ###
            self.cur.copy_pose_from(self.get_last_frame())
            ###
            return False
        self.ref_kps_mtched, self.ref_mask = ref_kps_mtched, ref_mask

        if DEBUG:
            ref_masked_img = self._renderer.drawMatches(
                self.cur.img, cur_kp,
                reference_frame.img, reference_frame.kps,
                ref_kps_mtched
            )
            self._renderer.render(ref_masked_img)

            # key_points_prediction = self._renderer.drawPredictions(
            #     reference_frame, reference_frame.kps,
            #     self.cur, cur_kp,
            #     ref_kps_mtched
            # )
            # self._renderer.render(key_points_prediction)

        # estimate R, t, i.e., camera to world matrix

        # estimate camera motion and poses with depths of keypoints for monocular camera using
        # RandSacWrapper + linear implementation of epipolar equation solver

        if len(ref_kps_mtched) < 10:  # unlikely to invoke
            print("Not enough points!")
            ###
            self.cur.copy_pose_from(self.get_last_frame())
            ###
            return False

        try:
            R, t, mask = self.resolve_pose(cur_kp, reference_frame.kps, ref_kps_mtched)
        except Exception as e:
            masked_img = self._renderer.drawMatches(
                self.cur.img, cur_kp,
                masked_img, last_frame.kps,
                kps_mtched
            )

            self._renderer.render(masked_img)
            print(e)
            print("Skip the frame ...")
            raise e
            return False

        # Note, opencv are operating pixel movemnt tranformation!
        # [e1, e2, e3] * x1 (world) = [R|t]*[e1, e2, e3]*x2, i.e. x1 = [R\t] * x2 (camera)

        # camera movement Twc
        Rcw = R
        tcw = t

        Rwc = np.linalg.inv(Rcw)
        twc = -Rwc.dot(t)

        # done
        if self.cur.timestamp is not None:
            R_gt = self.cur.R_gt[0:3, 0:3].dot(np.linalg.inv(reference_frame.R_gt[0:3, 0:3]))
            t_gt = self.cur.t_gt - R_gt.dot(reference_frame.t_gt)
            print("")
            print("Pose Ground truth (from T(#%d): %s to T(#%d): %s) \nR:\n%s\nt:\n%s\n" % (
                self.cur.pre.assoc_timestamp,
                self.cur.pre.seq,
                self.cur.assoc_timestamp,
                self.cur.seq,
                R_gt,
                t_gt
            ))

            print("Estimated pose (E/F, T: %s) \nR:\n%s\nt:\n%s\n" % (
                self.cur.timestamp,
                Rwc,
                twc
            ))

        else:
            print("Estimated pose (E/F) R:\n%s\n" % Rwc)
            print("Estimated pose (E/F) t:\n%s\n" % twc)

        if USE_POSE_GROUND_TRUTH and self.cur.timestamp is not None:
            print("Using pose ground truth instead")

            Rwc = R_gt
            twc = t_gt

            Rcw = np.linalg.inv(Rwc)
            tcw = -Rcw.dot(twc)
            pass

        anchor_point = Rwc.dot(reference_frame.t0) + twc
        # Later we will use device to populate
        camera = Camera(self._map.device, Rwc, twc, anchor_point, H=self.cur.img.shape[0], W=self.cur.img.shape[1])
        camera.Rcw = Rcw
        camera.tcw = tcw

        self.cur.set_camera(camera)
        # check solver relative epipolar contraint precision
        if DEBUG:
            t_mat = camera.t_SE3ToR3()
            K = camera.K
            E = t_mat.dot(Rwc)
            ComulativeErr = 0.
            AvgErr = 0.
            num_test_cases = len(ref_kps_mtched)
            logging.info("check solver relative epipolar constraint precision, test cases %d" % num_test_cases)
            for i, mtch in enumerate(ref_kps_mtched):
                x1 = camera.reproj(reference_frame.kps[mtch.trainIdx]).data.reshape(3, 1)
                x2 = camera.reproj(cur_kp[mtch.queryIdx]).data.reshape(3, 1)
                epipolar_constrain_eq = x2.T.dot(E.dot(x1))
                print("Epipolar contrain eqution <%d(cur), %d(reference_frame #%d)>: %f" % (
                    mtch.queryIdx, mtch.trainIdx, reference_frame.seq, epipolar_constrain_eq))
                ComulativeErr += epipolar_constrain_eq

            AvgErr = ComulativeErr / num_test_cases
            print("Avervage epipolar constrain error : %f" % AvgErr)

        # check computed pose with ground truth (Optional)

        # info fusion with IMU for a high accuracy R, t
        # @todo TODO

        # upate frame matrix stack
        self.cur.R0 = Rwc.dot(reference_frame.R0)
        self.cur.t0 = Rwc.dot(reference_frame.t0) + twc

        self.cur.R1 = Rwc
        self.cur.t1 = twc

        self.cur.Rcw = Rcw
        self.cur.tcw = tcw

        camera.R0 = self.cur.R0
        camera.t0 = self.cur.t0

        # change to bitwise operation
        try:
            grid_mask = self._renderer.drawGrid(self.cur)
            masked_img = cv2.add(masked_img, grid_mask)
        except Exception as e:
            print(e)
            pass

        try:
            axis_mask = self._renderer.drawAxis(self.cur)
            masked_img = cv2.add(masked_img, axis_mask)
        except Exception as e:
            print(e)
            pass

        masked_img = self._renderer.drawMatches(
            self.cur.img, cur_kp,
            masked_img, last_frame.kps,
            kps_mtched
        )

        self._renderer.render(masked_img)

        # decide whether we discard the frame and move on
        # check pixel disparity

        # @todo : TODO
        if self.cur.seq - reference_frame.seq < 5:
            return False

        # if there are enough kp for triangulation
        if self.checkDisparity():
            self._map.add_new_key_frame(self.cur)
            self.cur.isKeyFrame = True
            return True
        else:
            return False
        pass

    def checkDisparity(self):
        # @todo : TODO
        return True

    # later we will implement it using ROS alike pubsub services
    def _init_relocalization_serv(self):
        # @todo : TODO
        if len(self.mapblocks) - 1 < self.mapblock_idx:
            raise Exception("number of map blocks should be larger than 0!")

        mapblock = self.mapblocks[self.mapblock_idx]
        self.mapblock_idx += 1

        # send request to relocalizer
        self.relocalizer.add(mapblock)
        pass

    # Relocalization ALgorithm 1 : See SVSO report for details
    def _Relocalization(self):
        # check wehter the map share enough landmarks with old maps
        # if there are enough matches, we will lock and send the current map
        # to the remote to computed

        ref_block = self.mapblocks[self.mapblock_idx - 1]
        legacy_trackList = ref_block.trackList()

        active_trackList = self._map.trackList()

        print("[Tracker._Relocalization] legacy mapblock points: %d, active mapblock points: %d" %
              (
                  len(ref_block.keypointCloud), len(self._map.keypointCloud)))

        #
        matcher = ROIMatcher()
        # setting threshold to a larger one
        matcher.THR = 0.95

        N = len(legacy_trackList)
        M = len(active_trackList)

        # solving N*M assignment matrix using KM algorithm
        mtched_indice, unmtched_landmarks_indice, unmtched_detections_indice = matcher.mtch(legacy_trackList,
                                                                                            active_trackList,
                                                                                            product="feature_only")

        print("[Tracker._Relocalization] %d mtches, %d unmtched landmarks, %d unmtched detections" %
              (len(mtched_indice), len(unmtched_landmarks_indice), len(unmtched_detections_indice)))

        matched = len(mtched_indice)
        if matched < 5:
            print("[Tracker._Relocalization] Not enough matched (%d) !" % matched)
            print("[Tracker._Relocalization] skipping ...")
        else:
            print("[Tracker._Relocalization] %d matched found!" % matched)

            mtched = []
            unmtched_detections = []
            for match in mtched_indice:
                roi = legacy_trackList[match[0]]
                new_landmark = active_trackList[match[1]]

                print("legacy mapblock landmark %s(%d mappoints) +----> active mapblock roi %s(%d mappoints)" %
                      (roi, len(roi.points), new_landmark, len(new_landmark.points)))

                l1 = len(new_landmark.points)
                l2 = len(roi.points)
                if l1 == 0 or l2 == 0:
                    continue

                # if l1 >= 2*l2:
                #   continue

                if l1 * 2 <= l2:
                    continue

                pair = [new_landmark, roi]

                mtched.append(pair)

            if len(mtched) < 5:
                # print("[Tracker._Relocalization] Not enough valid matches (%d)!" % len(mtched))
                # print("[Tracker._Relocalization] skipping finally ...")

                mtched = {}
                unmtched_detections = []
                # == Merge landmarks with the same labels
                for match in mtched_indice:
                    roi = legacy_trackList[match[0]]
                    new_landmark = active_trackList[match[1]]

                    label = roi.label
                    if mtched.get(label, None) is None:
                        pair = {}
                        mtched[label] = pair
                        pair["roi"] = roi.clone()
                        pair["new_landmark"] = new_landmark.clone()
                        continue

                    pair = mtched[label]
                    pair["roi"].merge(roi)
                    pair["new_landmark"].merge(new_landmark)
                # ==

                mtched_array = []
                totalNumOfPoints = 0

                for k, pair in mtched.items():
                    roi = pair["roi"]
                    new_landmark = pair["new_landmark"]

                    print(
                        "legacy mapblock landmarks union %s(%d mappoints) +----> active mapblock roi union %s(%d mappoints)" %
                        (roi, len(roi.points), new_landmark, len(new_landmark.points)))

                    l1 = len(new_landmark.points)
                    l2 = len(roi.points)

                    if l1 == 0 or l2 == 0:
                        continue

                    # if l1 >= 2*l2:
                    #     continue

                    if l1 * 2 <= l2:
                        continue

                    totalNumOfPoints += l1
                    mtched_array.append([new_landmark, roi])

                if len(mtched_array) < 2 or totalNumOfPoints < 1000:
                    print("[Tracker._Relocalization] Not enough valid matches (%d) or points (%d)!" % (
                        len(mtched_array), totalNumOfPoints))
                    print("[Tracker._Relocalization] skipping finally ...")
                    return
                else:
                    print("[Tracker._Relocalization] %d valid matches and %d point cloud found!" % (
                        len(mtched_array), totalNumOfPoints))

                for unmatch_idx in unmtched_landmarks_indice:
                    roi = legacy_trackList[unmatch_idx]
                    unmtched_detections.append(roi)

                # do copy or using write locks
                with self._state_modifier_lock:
                    self.is_relocalization_mode = True
                    pass
                self.relocalizer.add(self._map, matched=mtched_array, unmtched_roi=unmtched_detections)
                return
                pass
            else:
                print("[Tracker._Relocalization] %d valid matches found!" % len(mtched))

            for unmatch_idx in unmtched_landmarks_indice:
                roi = legacy_trackList[unmatch_idx]
                unmtched_detections.append(roi)

            # do copy or using write locks
            with self._state_modifier_lock:
                self.is_relocalization_mode = True
                pass
            self.relocalizer.add(self._map, matched=mtched, unmtched_roi=unmtched_detections)

    def _PnP(self, isSucc):
        points, measurements, slidingWindows = fetchByRoI(self.cur, self._map)

        # see ORBSlam
        if len(points) < 15:
            print("Not enough points for PnP solver!")
            return False
        else:
            print("%d matched points found!" % len(points))

        # solve PnP problem using PoseGraph optimization
        self.cur.copy_pose_from(self.get_last_frame())

        solver = PnPSolver(self.cur, self._map)
        # solve!
        R, t = solver.solve(points, measurements)

        # @todo : TODO PoseGraph
        optimizer = PoseOptimization()
        optimizer.set_FromMap(self._map)
        optimizer.set_FromFrame(self.cur)
        optimizer.set_FromPoints(points)
        optimizer.set_FromMeasurements(measurements)

        # construct the graph
        optimizer.Init()

        # optimize!
        optimizer.optimize(max_iterations=3)

        # update refined pose
        self.cur.update_pose(optimizer.get_pose("Frame", self.cur.seq))

        #
        if self.cur.timestamp is not None:
            print("")
            print("previous frame pose \nR:\n%s\nt:\n%s\n" % (
                self.cur.pre.R0,
                self.cur.pre.t0
            ))

            print("Pose Ground truth \nR:\n%s\nt:\n%s\n" % (
                self.cur.R_gt,
                self.cur.t_gt
            ))

            print("Pnp resolved pose \nR:\n%s\nt:\n%s\n" % (
                R,
                t
            ))

            print("Pose graph optimization resolved pose \nR:\n%s\nt:\n%s\n" % (
                self.cur.R0,
                self.cur.t0
            ))

            pass

        if USE_POSE_GROUND_TRUTH:
            print("Using pose ground truth instead")
            R = self.cur.R_gt[0:3, 0:3]
            t = self.cur.t_gt
            pose_gt = g2o.SE3Quat(R, t.reshape(3, ))
            self.cur.update_pose(pose_gt)
            pass

        # at least there are two frames
        reference_frame = self._map.get_active_frames()[-1]
        #
        if self.last_frame is not None and reference_frame is not self.last_frame:
            # try to find more points with reference frame
            # might be running in local mapping thread to find new points

            # @todo : TODO generate more points, you can also do it in local mapping thread, see LocalMapping thread implementation

            pass

        pass

    # @todo : TODO
    # Cold start may take few seconds for visual only system. If IMU we could have more robust initial esitmation
    # of camera motions and poses.
    def _coldStartTrack(self):

        #
        cur_kp, cur_kp_features = self.cur.kps, self.cur.kps_feats
        cur_roi_kp, cur_roi_features = self.cur.roi_kp, self.cur.roi_features

        last_frame = self.ref  # self.get_last_frame()

        # matcher = self.cur.matchers['OpticalFlowBasedKeyPointsMatcher']
        # kps_mtched, mask = matcher.matches, matcher.mask.tolist()

        kps_mtched, mask = self.ref_kps_mtched, self.ref_mask
        camera = self.cur.camera

        # estimate keypoints depth and compute projection errors
        points, cur_cam_pts, last_cam_pts = self.triangulate(self.cur, last_frame, camera.R0, camera.t0, cur_kp,
                                                             last_frame.kps, kps_mtched)

        l = len(points)
        zdepth = []
        for i in range(l):
            mtch = kps_mtched[i]
            last_projection = last_frame.kps[mtch.trainIdx].pt
            cur_projection = cur_kp[mtch.queryIdx].pt

            # get world point
            p = points[i]

            if self.cur.depth_img is not None:
                depth_img = self.cur.depth_img

                SCALING_FACTOR = 5

                K = self.cur.camera.K
                x, y = cur_projection
                Z = depth_img[int(y), int(x)]
                Z /= SCALING_FACTOR
                if Z == 0:
                    p.setBadFlag()
                    continue
                X = (x - K[0, 2]) * Z / K[0, 0]
                Y = (y - K[1, 2]) * Z / K[1, 1]

                # p.update(X, Y, Z)

                # Twc
                v = np.array([X, Y, Z]).reshape((3, 1))
                v = self.cur.R0.dot(v) + self.cur.t0
                v = v.reshape((3,))

                # print("[Tracker._ColdStartTrack] Updating from %s to (%f, %f, %f)" % (
                #     p,
                #     v[0],
                #     v[1],
                #     v[2]
                # ))
                p.update(*v)
                pass

            # transform to camera space
            v = p.data.reshape(3, 1)
            v = np.linalg.inv(last_frame.R0).dot(v - last_frame.t0)

            z = v[2][0]

            # Rule #1
            if z < 0:
                print("%s's depth in Frame#%d's camera is negative! Set it to be bad point." % (p, self.cur.seq))
                p.setBadFlag()
            # Rule #2
            elif z > 100:
                print("%s's depth in Frame#%d's camera is larger than 100! Set it to be bad point." % (p, self.cur.seq))
                p.setBadFlag()
            else:
                zdepth.append(z)

            if DEBUG:
                isPlot = i % 100 == 0
                if isPlot:
                    logging.info("check %d point's reprojection error ... " % i)
                # from svso.py_pose_graph.point3d import Point3D # we don't have to increase counter here
                ## LAST
                p_last_camera = Point3D(v[0][0] / z, v[1][0] / z, 1)
                if i % 100 == 0:
                    print("p_last_camera: \n%s\n" % p_last_camera)

                dp1 = Point3D(
                    p_last_camera.x - last_cam_pts[i].x,
                    p_last_camera.y - last_cam_pts[i].y
                )

                if isPlot:
                    print("last frame (p) reproj err: (%f, %f), depth: %f" % (dp1.x, dp1.y, z))

                v = p.data.reshape(3, 1)

                cam_pt = np.linalg.inv(last_frame.R0).dot(v - last_frame.t0)

                gt_z = cam_pt[2][0]
                gt_p1 = Point3D(cam_pt[0][0] / gt_z, cam_pt[1][0] / gt_z, 1)
                if isPlot:
                    print("cam_pt(last frame): \n%s\n" % gt_p1.data.reshape((3, 1)))
                gt_p1.seq = p.seq

                gt_dp1 = Point3D(
                    gt_p1.x - last_cam_pts[i].x,
                    gt_p1.y - last_cam_pts[i].y
                )
                if isPlot:
                    print("[GT] last frame (p) reproj err: (%f, %f), depth: %f" % (gt_dp1.x, gt_dp1.y, gt_z))

                proj_pt_last_cam = last_frame.camera.view(gt_p1)
                if proj_pt_last_cam is None:
                    p.setBadFlag()
                    continue
                gt_dpx = Pixel2D(
                    proj_pt_last_cam.y - last_projection[1],
                    proj_pt_last_cam.x - last_projection[0]
                )

                # just for test
                if isPlot:
                    print("[GT] proj_pt_last_cam: \n%s\n" % proj_pt_last_cam)
                    print("[GT] last frame (p) projection err: (%f, %f), depth: %f" % (gt_dpx.x, gt_dpx.y, gt_z))

                ## CURRENT
                p_cur_camera = self.cur.Rcw.dot(v) + self.cur.tcw

                z1 = p_cur_camera[2][0]
                p_cur_camera = Point3D(p_cur_camera[0][0] / z1, p_cur_camera[1][0] / z1, 1)
                if isPlot:
                    print("p_cur_camera: \n%s\n" % p_cur_camera)

                dp2 = Point3D(
                    p_cur_camera.x - cur_cam_pts[i].x,
                    p_cur_camera.y - cur_cam_pts[i].y
                )

                if isPlot:
                    print("cur frame (p) reproj err: (%f, %f), depth: %f" % (dp2.x, dp2.y, z1))

                # just for test
                # cam_pt = self.cur.camera.viewWorldPoint(p)
                v = p.data.reshape(3, 1)

                cam_pt = np.linalg.inv(self.cur.R0).dot(v - self.cur.t0)
                # cam_pt = self.cur.R0.dot(v) + self.cur.t0

                gt_z = cam_pt[2][0]
                gt_p2 = Point3D(cam_pt[0][0] / gt_z, cam_pt[1][0] / gt_z, 1)
                if isPlot:
                    print("cam_pt(cur frame): \n%s\n" % gt_p2.data.reshape((3, 1)))
                # used for debug
                gt_p2.seq = p.seq

                gt_dp2 = Point3D(
                    gt_p2.x - cur_cam_pts[i].x,
                    gt_p2.y - cur_cam_pts[i].y
                )
                if isPlot:
                    print("[GT] cur frame (p) reproj err: (%f, %f), depth: %f" % (gt_dp2.x, gt_dp2.y, gt_z))

                proj_pt_cur_cam = self.cur.camera.view(gt_p2)
                if proj_pt_cur_cam is None:
                    # raise Exception("Wrong value!")
                    p.setBadFlag()
                    continue
                gt_dpx = Pixel2D(
                    proj_pt_cur_cam.y - cur_projection[1],
                    proj_pt_cur_cam.x - cur_projection[0]
                )
                # just for test
                if isPlot:
                    print("[GT] proj_pt_cur_cam: \n%s\n" % proj_pt_cur_cam)
                    print("[GT] cur frame (p) projection err: (%f, %f), depth: %f" % (gt_dpx.x, gt_dpx.y, gt_z))
                    print("\n\n")
                    print("====================================================================================")

            last_cam_pts[i].world = p
            cur_cam_pts[i].world = p
            # the point has not been associated with current frame and map block
            # we will see whether the pair of the point and its projection (p[i], last_cam_pts[i]) has already been
            # registered by the map block
            # p.update(v[0][0], v[1][0], v[2][0])

        # fill current structure
        # self.cur.fill()

        # compute median depth
        zdepth.sort()
        depthMedian = 1.
        if len(zdepth) > 0:
            depthMedian = float(zdepth[int(len(zdepth) / 2)])
        self.cur.medianSceneDepth = depthMedian

        #
        if not USE_POSE_GROUND_TRUTH:
            # trick when pose ground truth is not available
            # scale will be optimized in loopclosing SE3 optimization module
            camera.t1 /= depthMedian
            camera.t0 /= depthMedian

            #
            self.cur.t1 /= depthMedian
            self.cur.t0 /= depthMedian

            # scale points
            for p in points:
                if p.isBad():
                    continue
                p.x /= depthMedian
                p.y /= depthMedian
                p.z /= depthMedian

        # register KeyPoints
        self._map.registerKeyPoints(self.cur, cur_kp, last_frame, last_frame.kps, cur_cam_pts, last_cam_pts, points,
                                    kps_mtched, mask)

        # culling
        self._map.Culling()

        # refine poses and location of landmarks (by points)
        optimizer = BundleAdjustment().set_FromMap(self._map)
        optimizer.Init()

        print('num vertices:', len(optimizer.vertices()))
        print('num edges:', len(optimizer.edges()))

        # optimize!
        optimizer.optimize()

        # update refined coordinates and poses
        rsmes1 = optimizer.EstimateError()
        logging.info("Refined pose graph coordinates RSME error after Bundle Adjsutment: %f" % rsmes1[1])

        # optimizer.UpdateMap()

        pointCloud = self._map.trackPointsList()

        #
        #     if not USE_POSE_GROUND_TRUTH:
        #       # scale the computed map
        #       for _, p in pointCloud.items():
        #         if p.isBad():
        #           continue
        #         p.x /= depthMedian
        #         p.y /= depthMedian
        #         p.z /= depthMedian

        #       self.cur.t0 /= depthMedian
        #       self.cur.t1 /= depthMedian

        #       self.cur.camera.t0 /= depthMedian
        #       self.cur.camera.t1 /= depthMedian

        self._map.set_complete_updating()
        self.set_initialized()
        logging.info("The system has been initialized!")

    # @todo : TODO
    def get_reference_frame(self):
        if self.cur.is_First:
            return None

        frames = self._map.get_active_frames()
        if len(frames) - self._map.slidingWindow > 0:
            return frames[-1]
        else:
            # no frames now
            return None

    # @todo : TODO
    def get_last_frame(self):
        if self.cur.is_First:
            return None

        frames = self._map.get_frames()
        if len(frames) <= 1:
            return None
        else:
            return frames[-2]

    # @todo : TODO
    def collect_enough_kp_mtches(self):
        return False

    # @todo : TODO
    # see the implementation of ORBSlam for refernce, I simply borrow the name from it
    def track(self, img, timestamp=None):
        frame = Frame().set_FromImg(img)
        if timestamp is not None:
            frame.set_timestamp(timestamp)

            # query ground truth
            mtched = self.trajectories_gt.query_timestamp(timestamp, max_diff=0.01)

            t_gt = np.array(mtched[1:4]).reshape((3, 1))
            qut = Quaternion(*mtched[4:])
            R_gt = qut.Matrix4()

            frame.assoc_timestamp = mtched[0]
            frame.R_gt = R_gt
            frame.t_gt = t_gt

            # query depth
            depth_img_source = query_depth_img(self.depth_images, timestamp, max_diff=0.5)
            dest = os.path.join(TUM_DATA_DIR, depth_img_source)
            depth_img = cv2.imread(dest, 0)
            frame.depth_img = depth_img

        self.cur = frame
        self._map.add(frame)

        last_frame = self.get_last_frame()

        try:
            isSucc = self._match(last_frame)
        except RelocalizationErr as e:
            isSucc = False
            logging.info("switch state from %s to %s" % (self.state, Tracker.State.LOSTED))
            if self._is_relocalization_mode:
                print("Something wrong here ...")
            self.state = Tracker.State.LOSTED

        if not self.isInitialized():
            # to collect enough key frames
            if len(self._map.get_frames()) - self._map.slidingWindow > 0:
                self._init()

            # perform primitive tracker based on previous frames
            if isSucc:
                self._coldStartTrack()

            if self.isInitialized():
                self.REF_STEP = self.cur.seq - self.ref.seq
                self._map.slidingWindow = self.REF_STEP
                self.ref = self.cur
            return
        #
        else:
            if self.state == Tracker.State.INITIALIZED:
                logging.info("transition from state %s to %s " % (self.state, Tracker.State.TRACKING))
                self.state = Tracker.State.TRACKING

            # We already have registered 3d points and landmarks
            # Using them to compute campera poses with PnP solver
            if self.state == Tracker.State.TRACKING:
                logging.info("tracking ...")

                # try to construct pnp solver
                self._PnP(isSucc)

                self._map.add_new_key_frame(self.cur)

                #
                if isSucc:
                    # if we have enough matches
                    self.mapper.add(self.cur)
                    self.cur.isKeyFrame = True
            else:
                if self.state == Tracker.State.LOSTED:
                    # we cannot find enough matches with the last frame

                    # save the curren map block
                    self.mapblocks.append(self._map)

                    with self._state_modifier_lock:
                        self._is_relocalization_mode = True

                    mapblock = RuntimeBlock()
                    mapblock.set_device(self._map.device)

                    self.Reset_Relocalization()
                    self.set_FromMap(mapblock)
                    # self.Init()

                    # send request to relocalization module
                    # note we are not doing this step in main thread
                    # send request to Relocalization thread to activate the service
                    print("Init relocalization service")
                    self._init_relocalization_serv()
                    return
                else:
                    raise Exception("Not Implemented Yet!")
            pass

        if self._is_relocalization_mode:
            self._Relocalization()
            pass

        # update reference
        if self.cur.seq - self.ref.seq > 10:
            self._map.add_new_key_frame(self.cur)
            self.cur.isKeyFrame = True

            print("Setting %s to keyframe" % self.cur)
            # self.REF_STEP = self.cur.seq - self.ref.seq
            self._map.slidingWindow = self.REF_STEP

            self.ref = self.cur

            # send the current frame to mapper to fill more points
            # self.mapper.add(self.cur)

    def resolve_pose(self, cur_kps, last_kps, matches1to2):
        R_E, t_E, mask_E, score_E = self.resolve_with_essential(cur_kps, last_kps, matches1to2)
        R_rets, t_rets, n_rets, mask_H, score_H = self.resolve_with_homography(cur_kps, last_kps, matches1to2)

        ratio = score_H / (score_H + score_E)

        if ratio > 0.5:
            best_sol = 0
            best_norm_z = np.abs(n_rets[best_sol][2, 0])
            for i, n in enumerate(n_rets):
                norm_z = np.abs(n[2, 0])
                if norm_z > best_norm_z:
                    best_norm_z = norm_z
                    best_sol = i
            print("Using Homography Estimation")
            return R_rets[best_sol], t_rets[best_sol], mask_H
        else:
            print("Using Essential Estimation")
            return R_E, t_E, mask_E

    def resolve_with_homography(self, cur_kps, last_kps, matches1to2):
        # intrinsic parameters
        K = self._map.device.K
        # see ost file for camrea settings, or filling data from K directly
        focal_length = (K[0, 0] + K[1, 1]) / 2.0
        principle_point = (K[0, 2], K[1, 2])

        p1, p2 = [], []

        print("num of cur frame keypoints: %d" % len(cur_kps))
        print("num of last frame keypoints: %d" % len(last_kps))
        print("num of mtches: %d" % len(matches1to2))

        # last frame
        for i, mtch in enumerate(matches1to2):
            p1.append(last_kps[mtch.trainIdx].pt)
            p2.append(cur_kps[mtch.queryIdx].pt)

        homography, inliers = cv2.findHomography(np.float32(p1), np.float32(p2),
                                                 method=cv2.FM_LMEDS)
        homography /= homography[2, 2]

        mask = inliers.astype(bool).flatten()

        _, R_list, t_list, normals = cv2.decomposeHomographyMat(homography, K)

        # filter out bad solutions
        # see https://stackoverflow.com/questions/57075719/filtering-out-false-positives-from-feature-matching-homography-opencv
        #     https://github.com/opencv/opencv/blob/master/modules/calib3d/src/homography_decomp.cpp
        possibleSolutions = cv2.filterHomographyDecompByVisibleRefpoints(R_list, normals,
                                                                         np.float32(p1)[mask].reshape((-1, 1, 2)),
                                                                         np.float32(p2)[mask].reshape((-1, 1, 2)))
        R_rets, t_rets, n_rets = [], [], []
        if possibleSolutions is not None:
            l = len(possibleSolutions)
        else:
            l = 0
        for i in range(l):
            idx = possibleSolutions[i][0]
            R_rets.append(R_list[idx])
            t_rets.append(t_list[idx])
            n_rets.append(normals[idx])

        score, mask_new = self.computeHomographyScore(homography, p1, p2, mask)
        return R_rets, t_rets, n_rets, mask_new, score

    def resolve_with_essential(self, cur_kps, last_kps, matches1to2):
        # intrinsic parameters
        K = self._map.device.K
        # see ost file for camrea settings, or filling data from K directly
        focal_length = (K[0, 0] + K[1, 1]) / 2.0
        principle_point = (K[0, 2], K[1, 2])

        p1, p2 = [], []

        print("num of cur frame keypoints: %d" % len(cur_kps))
        print("num of last frame keypoints: %d" % len(last_kps))
        print("num of mtches: %d" % len(matches1to2))

        # last frame
        for i, mtch in enumerate(matches1to2):
            p1.append(last_kps[mtch.trainIdx].pt)
            p2.append(cur_kps[mtch.queryIdx].pt)

        # @todo : TODO compute Essential or Homography
        # p1 = cv2.undistortPoints(np.expand_dims(np.float32(p1), axis=1), cameraMatrix=K, distCoeffs=None)
        # p2 = cv2.undistortPoints(np.expand_dims(np.float32(p2), axis=1), cameraMatrix=K, distCoeffs=None)

        # find fundmental matrix
        USE_FUNDAMENTAL = False
        if USE_FUNDAMENTAL:
            F, inliers = cv2.findFundamentalMat(np.float32(p1), np.float32(p2),
                                                method=cv2.FM_LMEDS)

            # @todo TODO
            # if there are many key points sampling from plane object,
            # find homography using linear algebra wrapped by RandSac

            # else compute estimated the Essential Matrix and decompose the matrix
            # to recover camera pose

            if F is None:
                print("No solution found using FM_LMEDS. Switch to Randsac method ...")
                F, inliers = cv2.findFundamentalMat(np.float32(p1), np.float32(p2),
                                                    method=cv2.FM_RANSAC)

            F0 = F
            if F is None:
                raise Exception("No solutions!")

            num_solutions = F.shape[0] / 3
            if num_solutions > 1:
                print("choose the first solution")
                F = F0[0:3, 0:3]
            estimated_E = np.dot(K.T, np.dot(F, K))
        else:
            estimated_E, inliers = cv2.findEssentialMat(np.float32(p1), np.float32(p2),
                                                        focal=focal_length, pp=principle_point,
                                                        method=cv2.FM_LMEDS, prob=0.999, threshold=3.0)
            estimated_E /= estimated_E[2, 2]

        # decomposition Essential Matrix
        mask = inliers.astype(bool).flatten()
        _, R, t, _ = cv2.recoverPose(estimated_E, np.array(p1)[mask], np.array(p2)[mask], K)
        t /= t.T.dot(t)

        # compute score_E
        score, mask_new = self.computeEssentialScore(estimated_E, K, np.array(p1), np.array(p2), mask)
        return R, t, mask_new, score

    def computeHomographyScore(self, H, p1, p2, mask):
        score = 0.
        th = 5.991

        invSigmaSquare = 1.
        cnt = 0

        mask_new = np.zeros(mask.shape)

        N = mask.shape[0]
        for i in range(N):
            if not mask[i]:
                continue
            p_L = p1[i]
            p_R = p2[i]

            isGood = True
            x1 = np.array([p_L[0], p_L[1], 1]).reshape((3, 1))
            x2 = np.array([p_R[0], p_R[1], 1]).reshape((3, 1))

            # H12*x2
            uvw1 = np.linalg.inv(H).dot(x2)
            uvw1[2] = 1. / uvw1[2]
            uvw1[0] *= uvw1[2]
            uvw1[1] *= uvw1[2]

            _dir = x1[:2] - uvw1[:2]
            dist1 = _dir.T.dot(_dir) * invSigmaSquare

            if dist1 > th:
                isGood = False
            else:
                score += th - dist1

            # H21*x1
            uvw2 = H.dot(x1)
            uvw2[2] = 1. / uvw2[2]
            uvw2[0] *= uvw2[2]
            uvw2[1] *= uvw2[2]

            _dir = x2[:2] - uvw2[:2]
            dist2 = _dir.T.dot(_dir) * invSigmaSquare

            if dist2 > th:
                isGood = False
            else:
                score += th - dist2

            if isGood:
                mask_new[i] = 1
                cnt += 1

        print("H score: %f / %d" % (score, cnt))
        return score, mask_new

    def computeEssentialScore(self, estimated_E, K, p1, p2, mask):
        K_inv = np.linalg.inv(K)
        F = K_inv.T.dot(estimated_E).dot(K_inv)

        score = 0.
        th = 3.841
        chiSquare = 5.991

        # see BundleAdjustemnt
        invSigmaSquare = 1.0
        cnt = 0

        mask_new = np.zeros(mask.shape)

        N = mask.shape[0]
        for i in range(N):
            if not mask[i]:
                continue
            p_L = p1[i]
            p_R = p2[i]

            isGood = True
            x1 = np.array([p_L[0], p_L[1], 1]).reshape((3, 1))
            x2 = np.array([p_R[0], p_R[1], 1]).reshape((3, 1))

            # reprojection with second image
            dx1 = F.dot(x1)

            cost_dist = dx1.T.dot(x2) / (np.linalg.norm(x2) + 1e-6)

            chiSquare1 = np.power(cost_dist, 2) * invSigmaSquare
            if chiSquare1 > th:
                score += 0
                isGood = False
            else:
                score += chiSquare - chiSquare1

            #
            dx2 = x2.T.dot(F)
            dx2 = dx2.T

            cost_dist = dx2.T.dot(x1) / (np.linalg.norm(x1) + 1e-6)
            chiSquare2 = np.power(cost_dist, 2) * invSigmaSquare

            if chiSquare2 > th:
                score += 0
                isGood = False
            else:
                score += chiSquare - chiSquare2

            if isGood:
                mask_new[i] = 1
                cnt += 1

        print("E score: %f / %d" % (score, cnt))
        return score, mask_new

    # @todo : TODO PnP Solver
    def refine_pose(self, R, t, cur):
        # see _PnP
        pass

    def triangulate(self, cur, last_frame, R, t, cur_kps, last_kps, matches):
        pose_matl = np.c_[last_frame.R0, last_frame.t0]
        pose_matr = np.c_[R, t]

        cam = cur.camera

        p1, p2 = [], []

        points = []
        cur_cam_pts = []
        last_cam_pts = []

        # last frame
        for i, mtch in enumerate(matches):
            p1.append(cam.reproj(last_kps[mtch.trainIdx].pt).data)
            p2.append(cam.reproj(cur_kps[mtch.queryIdx].pt).data)

        p1 = np.float32(p1)
        p2 = np.float32(p2)
        print("p1.shape", p1.T.shape)
        ret = cv2.triangulatePoints(pose_matl, pose_matr, p1.T[0:2], p2.T[0:2])
        # make it homogenous
        ret /= ret[3]

        l = ret.shape[1]
        x1 = pose_matl.dot(ret)
        x2 = pose_matr.dot(ret)
        for i in range(l):
            vec = ret[:, i].T
            # points.append(Point3D(vec[0], vec[1], vec[2]).set_FromType("world"))
            points.append(WorldPoint3D(vec[0], vec[1], vec[2]))
            # last_cam_pts.append(Point3D(x1[0,i] / x1[2,i], x1[1,i] / x1[2,i], 1))
            last_cam_pts.append(CameraPoint3D(p1[i][0], p1[i][1], 1))
            # cur_cam_pts.append(Point3D(x2[0,i] / x2[2,i], x2[1,i] / x2[2,i], 1))
            cur_cam_pts.append(CameraPoint3D(p2[i][0], p2[i][1], 1))

        return points, cur_cam_pts, last_cam_pts

    # @todo : TODO
    def isInitialized(self):
        return self._initialized

    def set_initialized(self):
        with self._state_modifier_lock:
            self._initialized = True
            self.state = Tracker.State.INITIALIZED


# Semantic vision suppported tracker for multiple obstacles
class SVSOTracker(Tracker):
    def __init__(self):
        super().__init__()
