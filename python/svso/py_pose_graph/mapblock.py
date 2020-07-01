import numpy as np
import threading

import logging
from svso.lib.log import LoggerAdaptor

_logger = logging.getLogger("mapblock")

from svso.py_pose_graph.frame import Frame
from svso.py_pose_graph.point3d import CameraPoint3D, WorldPoint3D
from svso.matcher import ROIMatcher
from svso.optimizers.bundle_adjust import g2o
from svso.config import Settings

settings = Settings()

# setting debug variable
DEBUG = settings.DEBUG

# Utilities
def push(stack, e):
    stack.append(e)


def pop(stack, e):
    return stack.pop()


class Device:
    def __init__(self):
        self.fx = None
        self.cx = None
        self.fy = None
        self.cy = None

        self.distortion = None
        self.image_size = None

    @property
    def K(self):
        return np.array([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ])


class RuntimeBlock:
    logger = LoggerAdaptor("RuntimeBlock", _logger)

    def __init__(self):
        self._frames = []

        self.device = Device()

        # detected landmarks from sfe
        self.landmarks = {}

        # key points
        self.keypointCloud = {}

        # active frames selected from dynamic sliding window
        self.slidingWindow = 7

        # active frames stack: see discussion https://github.com/raulmur/ORB_SLAM2/issues/872
        self.active_frames = []

        # ====
        # lock sharing between the main thread (producer) and one of LocalMapping threads or Relocalization threads (consumers) could update the map
        self.modifying_mapblock_locker = threading.Lock()

        # automic boolean variable, guarded by self.modifying_mapblock_locker
        self.complete_modifying = True  # default state

        #
        self.modifying_completion_condition = threading.Condition(self.modifying_mapblock_locker)

        # =====
        # lock sharing between thread send request to draw the mapblock and thread updating it
        self.completion_locker = threading.Lock()

        # atomic boolean variable, guarded by self.completion_locker
        self.complete_updating = False

        self.update_map_condition = threading.Condition(self.completion_locker)

    def set_complete_updating(self):
        with self.completion_locker:  # lock should be required even though the guared variable is atomic
            self.complete_updating = True
            self.update_map_condition.notify()

    def set_complete_modifying(self):
        with self.modifying_mapblock_locker:
            self.complete_modifying = True
            self.modifying_completion_condition.notify()

    def set_device(self, device):
        self.device.fx = device.fx
        self.device.fy = device.fy
        self.device.cx = device.cx
        self.device.cy = device.cy
        self.device.distortion = device.distortion
        return self

    def load_device(self, device_path):

        def parseCalibratedDevice(fn_yaml):
            import yaml
            ret = {}
            skip_lines = 1
            with open(fn_yaml) as f:
                f.readline()
                content = f.read()
                parsed = yaml.load(content, Loader=yaml.FullLoader)
                return parsed

        parsed = parseCalibratedDevice(device_path)
        parsed_camera = parsed["Camera"]

        self.device.fx = parsed_camera["fx"]
        self.device.fy = parsed_camera["fy"]
        self.device.cx = parsed_camera["cx"]
        self.device.cy = parsed_camera["cy"]
        self.device.distortion = parsed_camera["distortion"]
        return self

    def add(self, entity):
        if isinstance(entity, Frame):
            if len(self._frames) == 0:
                entity.mark_as_first()
                # self.add_new_key_frame(entity)
            self._frames.append(entity)
        else:
            raise ValueError("expect entity to be type of %s but found %s" % (str(self.__class__), str(type(entity))))

    def get_frames(self):
        return self._frames

    def findFrame(self, frame_key):
        try:
            # return self._frames[frame_key - 1]
            for frame in self._frames:
                if frame.seq == frame_key:
                    return frame
            raise Exception("Not found!")
        except Exception as e:
            print("map frames:", self._frames)
            print("query frame key:", frame_key)
            raise (e)

    # @todo : TODO
    def get_active_frames(self):
        # @todo : TODO modify active frames list

        return self.active_frames

    # @todo : TODO
    def add_new_key_frame(self, entity):
        if isinstance(entity, Frame):
            push(self.active_frames, entity)
        else:
            raise TypeError("expect type of entity to be Frame, but found %s" % type(entity))

    # landmarks registration
    def register(self, detection):
        self.landmarks[detection.seq] = detection

    # keypoints registration
    def registerKeyPoints(self, cur_frame, kps1, last_frame, kps2, cur_cam_pts3, last_cam_pts3, points, mtched, mask):
        H, W = cur_frame.img.shape[0:2]
        newly_add = []
        numOfMatched = 0

        for i, mtch in enumerate(mtched):
            left_idx = mtch.queryIdx
            right_idx = mtch.trainIdx

            # kp observed by left_px using triangulation
            kp = points[i]
            if kp.isBad():
                continue

            # x - columns
            # y - rows
            (x1, y1) = kps1[left_idx].pt
            (x2, y2) = kps2[right_idx].pt

            left_px_idx = int(y1 * W + x1)
            right_px_idx = int(y2 * W + x2)

            # retrieve stored key points
            left_px = cur_frame.pixels[left_px_idx]
            right_px = last_frame.pixels[right_px_idx]

            # check whether kp is already observed by right_px
            mappoint = right_px.findObservationOf(kp)
            if mappoint is not None:
                #
                numOfMatched += 1
                try:
                    left_px.add_camera_point(cur_cam_pts3[i])
                except AttributeError:
                    left_px.add(cur_cam_pts3[i])

                print("point %s is already observed by mappoint %s" % (kp, mappoint))

                try:
                    mappoint.associate_with(cur_frame, left_px_idx)
                except Exception as e:
                    print(e)

                    # exit(-1)
                    pass

                # processing ROI
                landmark = left_px.roi.findParent()
                try:
                    landmark.associate_with(mappoint, cur_frame, left_px_idx)
                except Exception as e:
                    print(e)

                    # exit(-1)
                    pass

                try:
                    landmark.associate_with(mappoint, last_frame, right_px_idx)
                except:
                    print(e)

                    # exit(-1)
                    pass
            else:
                #
                try:
                    left_px.add_camera_point(cur_cam_pts3[i])
                except AttributeError:
                    left_px.add(cur_cam_pts3[i])
                cur_cam_pts3[i].px = left_px
                cur_cam_pts3[i].world = kp

                try:
                    right_px.add_camera_point(last_cam_pts3[i])
                except AttributeError:
                    right_px.add(last_cam_pts3[i])
                last_cam_pts3[i].px = right_px
                last_cam_pts3[i].world = kp

                # upgrade from camera point to map key point
                try:
                    kp.associate_with(cur_frame, left_px_idx)
                    # print("Associate %s with #F%d.%s at <%d>" % (kp, cur_frame.seq, left_px, left_px_idx))
                    kp.associate_with(last_frame, right_px_idx)
                    # print("Associate %s with #F%d.%s at <%d>" % (kp, last_frame.seq, right_px, right_px_idx))
                except Exception as e:
                    print(e)
                    raise (e)

                # compute an uuid key for registration
                # This varies from one programe to another. For example, if a program
                # retrieves it from tiles, a key is computed by uint64(tileid) << 32 | seq_id

                key = self.get_point3d_key(cur_frame.seq, i)
                kp.key = key

                # set RGB color for rendering
                # color = cur_frame.img[int(y1), int(x1)]
                # kp.set_color(color)

                # store it in the map block instance

                # @todo : TODO octree check

                # store the point into a concurrent hash map. Note since in python an iterator
                # is guarded by Global Interpretor Lock (GIL) so that to be executed by only one thread, we can
                # directly use it but suffer from performance issues.
                self.keypointCloud[key] = kp
                newly_add.append(kp)

                # Processing ROI
                landmark = left_px.roi.findParent()
                landmark.associate_with(kp, cur_frame, left_px_idx)
                landmark.associate_with(kp, last_frame, right_px_idx)
                print("Associate KeyPoint %s with RoI %s" % (kp, landmark))
                landmark.add(kp)
                kp.set_color(np.array(landmark.color or [1., 1., 0.]) * 255.)

        if DEBUG:
            print("Newly added world points: %d; Points matched to point cloud: %d" % (len(newly_add), numOfMatched))

    def get_point3d_key(self, tile_id, seq_id):
        if tile_id is None:
            tile_id = 0.
        return tile_id << 32 | seq_id

    # @todo : TODO
    def track(self, frame, detected_objects):
        if frame.is_First:
            self.logger.info("Add %d detected objects to initialize landmarks" % len(detected_objects))
            for ob in detected_objects:
                self.landmarks[ob.seq] = ob
            return tuple()
        else:
            trackList = self.trackList()

            for landmark in trackList:
                landmark.predict(frame)

            matcher = None
            if frame.matchers.get('ROIMacher', None) is None:
                matcher = ROIMatcher()
                frame.matchers['ROIMacher'] = matcher

            N = len(trackList)
            M = len(detected_objects)

            # solving N*M assignment matrix using KM algorithm
            mtched_indice, unmtched_landmarks_indice, unmtched_detections_indice = matcher.mtch(trackList,
                                                                                                detected_objects)

            print("%d mtches, %d unmtched landmarks, %d unmtched detections" %
                  (len(mtched_indice), len(unmtched_landmarks_indice), len(unmtched_detections_indice)))

            mtched, unmtched_landmarks, unmtched_detections = [], [], []
            # mtched List<Tuple> of (row,col,weights[row,col],distance[row,col]))
            for match in mtched_indice:
                landmark = trackList[match[0]]
                detection = detected_objects[match[1]]

                detection.parent = landmark
                landmark.records.append((detection, detection.frame.seq, landmark.predicted_states))
                landmark.update(detection)

                mtched.append((landmark, detection))

            # mark unmtched_landmarks
            # @todo : TODO
            for idx in unmtched_landmarks_indice:
                landmark = trackList[idx]
                unmtched_landmarks.append(landmark)

            # add unmtched_detections to landmarks
            l = len(unmtched_detections_indice)
            if l > 0:
                self.logger.info("Adding %d new landmarks" % l)
                for j in unmtched_detections_indice:
                    detection = detected_objects[j]
                    if self.landmarks.get(detection.seq, None) is not None:
                        raise ValueError("The detection has already been registered")
                    self.landmarks[detection.seq] = detection

                    unmtched_detections.append(detection)
            else:
                self.logger.info("No new landmarks found.")

            # do something with the mtched
            if DEBUG:
                # rendering the matches
                pass

            return mtched, unmtched_landmarks, unmtched_detections

    def trackKeyPoints(self, cur_frame, kps, kps_features, last_frame=None, matches1to2=None):
        if cur_frame.is_First:
            self.logger.info("Add %d extracted keypoints to initialize key points cloud" % len(kps))
            H, W = cur_frame.img.shape[0:2]
            for i, kp in enumerate(kps):
                # create a world, though we don't know depth infomation and camera poses
                world = WorldPoint3D(-1, -1, -1)
                # just for demo, don't need to worry about it
                world.type = "world"
                # see cv2::KeyPoint for details
                x, y = kp.pt
                px_idx = int(y * W + x)
                px = cur_frame.pixels[px_idx]
                world.associate_with(cur_frame, px_idx)

                local = CameraPoint3D(-1, -1, -1)
                local.world = world

                local.px = px
                try:
                    px.add_camera_point(local)
                except AttributeError:
                    px.add(local)
        else:
            # @todo : TODO compare with map points directly
            raise Exception("Not Implemented Yet!")
            pass
        pass

    def trackList(self):
        _trackList = []
        for key, landmark in self.landmarks.items():
            if landmark.is_active() or landmark.viewable():
                _trackList.append(landmark)
        self.logger.info("Retrieve %d active and viewable landmarks" % len(_trackList))
        print(self.landmarks)
        return _trackList

    # @todo : TODO
    def trackPointsList(self):
        return self.keypointCloud

    def Culling(self):
        trackList = self.trackList()

        for landmark in trackList:
            landmark.culling()
        pass

    def UpdateMap(self, R, t):
        print("Updating matrix: \nR:\n%s\nt:\n%s\n" % (R, t))

        # Updating points coordinates
        pointCloud = self.keypointCloud
        for k, p in pointCloud.items():
            v = p.data.reshape((3, 1))
            v = R.dot(v) + t
            v = v.reshape((3,))
            #       self.logger.info("Updating points from %s to %s" % (
            #         p.data,
            #         v
            #       ))
            p.update(*v)

        # Updating poses
        frames = self.get_active_frames()

        for frame in frames:
            R0 = R.dot(frame.R0)
            t0 = R.dot(frame.t0) + t

            print("Updating pose: \nR:\n%s\nt:\n%s\n" % (R0, t0))

            pose = g2o.SE3Quat(R0, t0.reshape((3,)))
            frame.update_pose(pose)
            pass
        pass

    # @todo : TODO
    def Merge(self, other_map_block,
              matched_rois, unmtched_rois,
              matched_points, unmatched_points):

        other = other_map_block

        # add unseen points
        for k, p in unmatched_points.items():
            if self.keypointCloud.get(k, None) is not None:
                raise ValueError("Wrong Value!")
            self.keypointCloud[k] = p

        # add unseen landmarks
        for roi in unmtched_rois:
            if self.landmarks.get(roi.seq, None) is not None:
                raise ValueError("The roi has already been registered!")
            self.landmarks[roi.seq] = roi

        return self