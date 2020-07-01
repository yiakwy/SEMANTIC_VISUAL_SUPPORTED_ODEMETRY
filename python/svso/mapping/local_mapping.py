import numpy as np
from svso.py_parallel_tasking_sched.threadingpool import ThreadingPool, Worker, RuntimeErr, queue
from svso.optimizers.bundle_adjust import LocalBA
from svso.localization.pnp import fetchByRoI

import logging
from svso.log import LoggerAdaptor as LogAdaptor

_logger = logging.getLogger("local_mapping")

# Local Mapping Worker. Note it is possible to be executed in a threading pool as it should be
class LocalMapping(Worker):

    logger = LogAdaptor(_logger, "LocalMapping")

    def __init__(self):
        tasks = queue.Queue()
        super().__init__(tasks, name="localmapping")

        # == Identity

        # == Thread local variables, and will not be touched by other threads (main thread) directly

        # used for local BA
        # self._frames_lock = threading.Lock()
        self._frames = []
        #
        self._optimizer = None
        # set in host
        self._map = None

        #
        self._tracker = None

        #
        self.cur = None

        #
        self._reference = None

    def set_stopping(self):
        # drift out task queue
        self.tasks.join()
        # close the while true loop
        with self._lock:
            self._stopping = True

    def set_FromMap(self, map):
        self._map = map
        return self

    def set_FromTracker(self, tracker):
        self._tracker = tracker
        return self

    def add(self, frame):
        self.tasks.put((None, frame, {}))
        return self

    # @todo : TODO
    def CreateMapPoints(self, cur_frame):
        reference_frame = self.get_reference_frame()

        cur_kp, cur_kp_features = cur_frame.kps, cur_frame.kps_feats

        # important!
        kps_mtched, mask = cur_frame.match(reference_frame, reference_frame.kps, reference_frame.kps_feats)

        # using R0, t0 computed from _PnP routine
        R = cur_frame.R0  # cur_frame.R0.dot(np.linalg.inv(reference_frame.R0))
        t = cur_frame.t0  # cur_frame.t0 - R.dot(reference_frame.t0)

        # triangulate the points
        points, cur_cam_pts, ref_cam_pts = self._tracker.triangulate(cur_frame, reference_frame, R, t, cur_kp,
                                                                     reference_frame.kps, kps_mtched)

        for i, p in enumerate(points):
            mtch = kps_mtched[i]
            cur_projection = cur_kp[mtch.queryIdx].pt
            if cur_frame.depth_img is not None:
                depth_img = cur_frame.depth_img

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

                # print("[LocalMapping.CreateMapPoints] Updating from %s to (%f, %f, %f)" % (
                #     p,
                #     v[0],
                #     v[1],
                #     v[2]
                # ))
                p.update(*v)
                pass
            pass

        with self._map.modifying_mapblock_locker:  # see PEP context manager protocal for details
            print("[Thread %d] LocalMapping.CreateMapPoints, waiting for peer modifiers to complete ..." % \
                  (self.seq,))
            self._map.modifying_completion_condition.wait_for(lambda: self._map.complete_modifying)
            print("[Thread %d] LocalMapping.CreateMapPoints, begin to modify mapblock." % \
                  (self.seq,))
            # will be roll back by "set_compelte_modifying"
            self._map.complete_modifying = False

            self._map.registerKeyPoints(cur_frame, cur_kp, reference_frame, reference_frame.kps, cur_cam_pts,
                                        ref_cam_pts, points, kps_mtched, mask)

            self._map.Culling()

            # used in the inner of a thread, no notification
            print("[Thread %d] LocalMapping.CreateMapPoints, compelte modifying mapblock." % \
                  (self.seq,))
            self._map.complete_modifying = True
        pass

    #
    def get_reference_frame(self):
        if self.cur.is_First:
            return None

        frames = self._map.get_active_frames()
        l = len(frames)
        if l > 0:
            if l - self._map.slidingWindow >= 0:
                return frames[l - self._map.slidingWindow]
            else:
                return frames[0]
        else:
            # no frames now
            return None

    def LocalBA(self):
        points, measurements, slidingWindows = fetchByRoI(self.cur, self._map)

        self._optimizer = LocalBA()
        self._optimizer.set_FromMap(self._map)
        self._optimizer.set_FromFrame(self.cur)
        self._optimizer.set_FromPoints(points)
        self._optimizer.set_FromMeasurements(measurements)

        # construct the graph
        self._optimizer.Init()

        # optimize!
        # len(adjusted_frames) == 0 will throw segmentation error!
        if len(self._optimizer.adjusted_frames) > 0:
            self._optimizer.optimize(max_iterations=7)

            # update coordinates
            with self._map.modifying_mapblock_locker:
                print("[Thread %d] LocalMapping.LocalBA, waiting for peer modifiers to complete ..." % \
                      (self.seq,))
                self._map.modifying_completion_condition.wait_for(lambda: self._map.complete_modifying)
                print("[Thread %d] LocalMapping.LocalBA, begin to modify mapblock." % \
                      (self.seq,))
                # will be roll back by "set_compelte_modifying"
                self._map.complete_modifying = False

                self._optimizer.UpdateMap()

                # used in the inner of a thread, no notification
                print("[Thread %d] LocalMapping.LocalBA, compelte modifying mapblock." % \
                      (self.seq,))
                self._map.complete_modifying = True

            print("[LocalMapping.LocalBA] done.")

        pass

    # override
    def _run_once(self):
        print("[Thread %d] fetching a key frame..." % \
              (self.seq,))
        _, frame, _ = self.tasks.get()
        print("[Thread %d] a new key frame %s fetched." % \
              (self.seq, frame))
        try:
            print("[Thread %d] running local bundle adjust upon the new KeyFrame %s" % \
                  (self.seq, frame))

            # with self._frames_lock:
            self._frames.append(frame)

            #
            self.cur = frame

            self.CreateMapPoints(frame)

            # Local BA
            # I am not sure whether LocalBA change the coordinates in a wrong way
            self.LocalBA()

        except RuntimeErr as e:
            print(e) # self.logger.error(e)
        except Exception as e:
            print(e) # self.logger.error(e)
        finally:
            self.tasks.task_done()
            # notify the main thread to send drawing request to drawer
            self._map.set_complete_updating()
            print("[Thread %d] complete updating mapblock. Notify drawer." % \
                  (self.seq,))
            # notify the relocalization thread to update the map using computed R,t
            self._map.set_complete_modifying()
            print("[Thread %d] complete modifying mapblock. Notify Relocalizer" % \
                  (self.seq,))