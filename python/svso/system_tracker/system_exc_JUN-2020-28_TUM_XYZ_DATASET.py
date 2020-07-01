import sys
import os
import cv2

from svso.py_pose_graph.mapblock import RuntimeBlock
from svso.system_tracker.tracker import SVSOTracker, Tracker
from svso.graphics.viewer import PCViewer
from svso.validation_toolkit.tum import Program as GTReader
from svso.py_parallel_tasking_sched.threadingpool import ThreadingPool

from svso.lib.log import init_logging
import logging

init_logging()

from svso.config import Settings

settings = Settings()
print(settings)
Project_base = settings.PROJECT_ROOT

class System:

    def __init__(self):
        # construct a map block, where typically raw map data should be read from this point
        block = RuntimeBlock()
        self._block = block
        block.load_device("{project_base}/data/tum/camera1.yaml".format(project_base=Project_base))

        # initialize a tracker
        tracker = SVSOTracker().set_FromMap(block)
        self._tracker = tracker

        # reading ground truth
        # NOTE: modify "global_settings.py" to decide whether you want to use ground truth
        trajectories, timestamps, depth_images = GTReader()
        self._trajectories = trajectories
        self._timestamps = timestamps
        self._depth_images = depth_images

        # set ground truth we loaded before
        tracker.trajectories_gt = trajectories

        # set depth images
        tracker.depth_images = depth_images

        # viewer
        self._legacy_viewer = None

        self._viewer = PCViewer()
        viewer = self._viewer

        viewer.set_FromTracker(tracker)
        viewer.set_FromMap(block)
        # viewer.Init()
        # Pycharm has some problems to run this snippet of codes
        viewer.Start()
        print("[Main Thread] viewer: ", viewer)
        print("\n\n")
        pass

    def run(self):
        tracker = self._tracker
        timestamps = self._timestamps
        viewer = self._viewer

        DRAW_ONCE = False

        # ORBSLAM2 first successfully triangulated frame no: #91

        SAVER = settings.SAVER
        if not os.path.isdir(SAVER):
            os.makedirs(SAVER)

        # allocating system resources
        pool = ThreadingPool(5)

        # pool.add_task(viewer.Start)

        # experiments controls
        cnt = 0  # total frames seq
        cnt0 = 0  # tracked frames seq

        START_FRAMES = 0
        STOP_FRAMES = 500
        TRACKED_FRAMES = 400

        STEP = 5

        P1 = START_FRAMES
        P2 = START_FRAMES + 3 * STEP  # 3
        P3 = START_FRAMES + 50 * STEP  # 50

        capture = cv2.VideoCapture(os.path.join(settings.VIDEO_DIR, settings.VIDEO_NAME))

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            timestamp = float(timestamps[cnt][0])
            file_name = timestamps[cnt][1]

            cnt += 1
            if cnt % STEP != 1:
                # comment this line if you want track images continuously
                pass
                # continue

            if cnt < START_FRAMES:
                continue

            # Relocalization control
            if cnt > P2 and cnt < P3:
                # creating missing gaps
                logging.info("skiping seqences from %d to %d, cur %d" % (
                    P2, P3, cnt
                ))
                continue

            # Trigger relocalization mode
            if cnt == P3:
                logging.info("manually switch state")
                tracker.state = Tracker.State.LOSTED
                DRAW_ONCE = False
                # store the legacy
                self._legacy_viewer = viewer
                # create a new viewer
                self._viewer = PCViewer()
                viewer = self._viewer
                viewer.set_FromTracker(tracker)
                viewer.set_FromMap(tracker._map)
                # viewer.Init()
                # Pycharm has some problems to run this snippet of codes
                viewer.Start()

            # @todo : TODO fix encoding error
            logging.info("exec tracker to track motions: %d" % cnt)
            # Hybrid of OpticalFlow and Kalman Filter predictor & deep features based Hungarian algorithm implementation, Updated on Feb 26 2020 by Author Lei Wang
            # tracker.track(frame)
            tracker.track(frame, timestamp=timestamp)
            cnt0 += 1

            if tracker.cur is None:  # relocalization triggered!
                continue

            ### Deprecated codes, in favor of new implementation of WebImagaRenderer ###
            # offline task
            if tracker.cur is not None and (
                    not hasattr(tracker.cur, "rendered_img") or tracker.cur.rendered_img is None):
                logging.info("skipping rendered_img at Frame#%d ..." % tracker.cur.seq)
                continue

            if not tracker.cur.is_First and tracker.cur.isKeyFrame:
                if tracker.cur.depth_img is not None and not DRAW_ONCE:
                    viewer.drawDepthImage(tracker.cur)
                    DRAW_ONCE = True
                logging.info("Scheduling task of updating point cloud viewer at KeyFrame %s" % tracker.cur)

                def update_map(cv, cur=None, last=None, flow_mask=None, active_frames=None):
                    with cv:
                        logging.info("Waiting for completion of updating map ...")
                        cv.wait_for(
                            lambda: tracker._map.complete_updating)  # awaken by local mapper, and check complete_updating variable
                        logging.info("Updating ...")
                        viewer.Update(cur=cur, last=last, flow_mask=flow_mask, active_frames=active_frames)
                        logging.info("Point cloud viewer updated at KeyFrame %s" % tracker.cur)
                        tracker._map.complete_updating = False

                if not pool.tasks.full():
                    pool.add_task(update_map, tracker._map.update_map_condition,
                                  cur=tracker.cur, last=tracker.last_frame, flow_mask=tracker.flow_mask,
                                  active_frames=tracker._map.get_active_frames())
                else:
                    logging.info("Tasks are full. Cannot push tasks into the pool.")

            if not tracker.cur.is_First and not tracker.cur.isKeyFrame:
                # update pose
                viewer.Update()
                pass

            # rets = model.detect([tracker.last_frame.img], verbose=1)
            # r = rets[0]
            # visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
            # out_frame = save_instances(tracker.last_frame.img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

            name = os.path.join(settings.SAVER, "images/{}.jpg".format(cnt))
            if tracker.last_frame is None or not hasattr(tracker.last_frame,
                                                         "rendered_img") or tracker.last_frame.rendered_img is None:
                out_frame = tracker.cur.rendered_img
            else:
                out_frame = cv2.add(tracker.last_frame.rendered_img, tracker.flow_mask)
            cv2.imwrite(name, out_frame)

            if cnt >= STOP_FRAMES:
                logging.info("break after %d frames" % STOP_FRAMES)
                break

            if cnt0 >= TRACKED_FRAMES:
                logging.info("break after %d frames tracked" % TRACKED_FRAMES)
                break

            if tracker.isInitialized():
                continue

        logging.info("complete reading video.")
        capture.release()
        # viewer.Stop()
        pass


if __name__ == "__main__":
    system = System()
    system.run()