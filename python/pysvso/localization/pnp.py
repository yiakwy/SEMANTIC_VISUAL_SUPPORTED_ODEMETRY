import cv2
import numpy as np

# used to evaluated pose graph optimization inside PnP Solver
class Measurement:

    def __init__(self, pt3d, frame1, px2d1, frame2, px2d2):
        self.pt3d = pt3d
        #
        self.frame1 = frame1
        self.px2d1 = px2d1
        self.frame2 = frame2
        self.px2d2 = px2d2

    def __str__(self):
        return "<Measurement p(%d)-> [Frame#%d, Frame%d]>" % (
            self.pt3d.seq,
            self.frame1.seq,
            self.frame2.seq
        )


# available method: projection | BF
def fetchByRoI(cur_frame, mapblock, method="projection"):
    detections = cur_frame._detections
    detected_rois = [] if detections is (None, list) else [obj.findParent() for obj in detections]
    frames = mapblock.get_frames()

    slidingWindows = set()

    #
    kps, kp_feats = cur_frame.kps, cur_frame.kps_feats
    img_shp = cur_frame.img.shape[0:2]

    # init feat_map
    feat_map = np.full(img_shp, -1)
    for i, kp in enumerate(kps):
        x, y = kp.pt
        if int(y) >= img_shp[0] or int(y) < 0 or \
                int(x) >= img_shp[1] or int(x) < 0:
            continue
        feat_map[int(y), int(x)] = i

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

        if x2 >= W:
            x2 = W - 1
        if y2 >= H:
            y2 = H - 1

        try:
            indice = feat_map[y1:y2, x1:x2] != -1
        except Exception as e:
            print(e)
            print("Slicer(y1, y2, x1, x2) : Slicer(%d, %d, %d, %d)" % (y1, y2, x1, x2))
            raise e
        return feat_map[y1:y2, x1:x2][indice]

    # retrieve mappoints observed
    points = {}
    measurements = {}

    print("Begin to fetch matched points for PnP ...")
    print("Total fetched ROIs: %d" % len(detected_rois))

    # fetch points visible to the last frame
    camera = cur_frame.pre.camera
    # backup codes
    if camera is None:
        # find a nearest camera to use
        cf = cur_frame.pre
        while camera is None:
            cf = cf.pre
            if cf is not None:
                camera = cf.camera
            else:
                break

    for roi in detected_rois:
        pointCloud = roi.points
        print("Total fetched points for RoI %s: %d" % (roi, len(pointCloud)))
        for _, point in pointCloud.items():
            # print("viewing point %s using camera from Frame#%d" % (point, cur_frame.pre.seq))

            cam_pt = camera.viewWorldPoint(point)

            projection = camera.view(cam_pt)
            if projection is None:
                continue

            if points.get(point.seq, None) is not None:
                continue

            # @todo : TODO get predicted projection with OpticalFlowKPntPredictor
            row = projection.y  # + flow[projection.x, projection.y][1]
            col = projection.x  # + flow[projection.x, projection.y][0]

            # set searching radius to a large number
            indice = _get_neighbors(50, int(row), int(col), feat_map, img_shp)
            if len(indice) == 0:
                continue

            # we can see the point
            points[point.seq] = point
            # print("Adding %s point" % point)
            # print("The point has been observed by %d frames" % len(point.frames))

            # for frame_key, pixel_pos in point.frames.items():
            for frame_key, pixel_pos in point.observations.items():
                # frame = frames[frame_key - 1]
                frame = mapblock.findFrame(frame_key)
                px_l = frame.pixels.get(pixel_pos, None)
                if px_l is None:
                    print("Could not find %s projection at Frame#%d pixel location %d" % (
                        point,
                        frame_key,
                        pixel_pos
                    ))
                    print(frame)
                    raise Exception("Unexpected Value!")
                feat_l = px_l.feature

                # KNN Search
                dist = None
                min_dist, min_ind = np.inf, None

                for ind in indice:
                    feat_r = kp_feats[ind]
                    dist = _hamming_distance(feat_l, feat_r)
                    if min_dist > dist:
                        min_dist = dist
                        min_ind = ind
                    pass  # indice

                # add to measurement
                x, y = kps[min_ind].pt
                H, W = img_shp[:2]
                px_r = cur_frame.pixels.get(int(y * W + x), None)
                if px_r is None:
                    raise Exception("Unexpected Value!")

                measurements[point.seq] = Measurement(point, frame, px_l, cur_frame, px_r)
                # print("Adding measurement for %s" % point)

                # update sliding window
                slidingWindows.add(frame_key)

                # associate with frame

                pass  # observations
            pass  # pointCloud
        pass  # detected_rois

    return points, measurements, list(slidingWindows)


# checking whether our results from PoseOptimization close to this, see Tracker._PnP (PoseOptimization) method
# defaults to cv2.solvePnPRansac
class PnPSolver:
    MIN_INLIERS = 10

    def __init__(self, frame, mapblock):
        self.frame = frame
        self._map = mapblock
        self._impl = cv2.solvePnPRansac
        self.inliers = None
        pass

    # @todo : TODO
    def solve(self, points, measurements):
        K = self.frame.camera.K

        pointCloud = []
        observations = []

        for _, point in points.items():
            pointCloud.append(point.data)
            measurement = measurements[point.seq]
            observations.append(measurement.px2d2.data)

        if len(pointCloud) < 6:
            print("Not Enough Points for PnP Solver!")
            return None, None

        try:
            _, rot_vec, tvec, inliers = self._impl(np.float32(pointCloud), np.float32(observations), K, None, None,
                                                   None,
                                                   False, 100, 4.0, 0.99, None)
        except Exception as e:
            print(e)
            return None, None

        R, _ = cv2.Rodrigues(rot_vec)
        t = tvec.reshape((3,1))
        if inliers is None or len(inliers) < self.MIN_INLIERS:
            print("inliners:", inliers)
            return None, None

        self.inliers = inliers
        R = np.linalg.inv(R)
        t = -R.dot(t)
        return R, t