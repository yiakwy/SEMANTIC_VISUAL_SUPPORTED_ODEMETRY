import numpy as np
from enum import Enum
from pysvso.py_parallel_tasking_sched.threadingpool import ThreadingPool, Worker, RuntimeErr, queue
from pysvso.lib.maths.rotation import Euler, Quaternion
from pysvso.optimizers.icp import ICP, NearestNeighbors
from pysvso.config import Settings

settings = Settings()

# setting debug variable
DEBUG = settings.DEBUG
USE_POSE_GROUND_TRUTH = settings.USE_POSE_GROUND_TRUTH

import logging
from pysvso.log import LoggerAdaptor as LogAdaptor

_logger = logging.getLogger("relocalization")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

def plot_data(X1, X2, Z, ax=None, color='b'):
  if ax is None:
      fig = plt.figure(dpi=60)
      ax = fig.gca(projection='3d')
  ax.scatter(X1, X2, Z, c=color, marker='o', s = 64)

  for i,x,y,z in zip(range(len(X1)), X1, X2, Z):
      p,q,_ = proj3d.proj_transform(x,y,z, ax.get_proj())
      # ax.annotate("%s" % i, xy=(p,q), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))#arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
      # ax.annotate("%s" % i, xy=(p,q))
      # ax.text(x,y,Z[i], '%s' % (str(i)), size=20, color='red')
  # plt.show()
  return ax


def user_plot_wrapper(*args, **kw):
    pcloud_dest = kw['dest']

    def _user_plot(pcloud):
        X1 = pcloud[:, 0]
        X2 = pcloud[:, 1]
        Z = pcloud[:, 2]

        ax = plot_data(X1, X2, Z)
        ax = plot_data(pcloud_dest[:, 0], pcloud_dest[:, 1], pcloud_dest[:, 2], ax=ax, color='r')

        plt.show()
        return ax

    return _user_plot


class Relocalization(Worker):

    logger = LogAdaptor(_logger, "Relocalization")

    class Algorithm(Enum):
        ICP = 1
        RANSAC_OVER_VOLUME = 2

    def __init__(self):
        tasks = queue.Queue()
        super().__init__(tasks, name="relocalization")

        # == Identity

        # == Thread local variables, and will not be touched by other threads (main thread) directly

        # self._maps_lock = threading.Lock()
        self._mapblocks = []

        #
        self.algorithm = Relocalization.Algorithm.ICP

        #
        self._optimizer = ICP()

        # set in host

        #
        self._tracker = None

        #
        self.cur_map = None

        #
        self.reference_map = None

    def set_stopping(self):
        # drift out task queue
        self.tasks.join()
        # close the while true loop
        with self._lock:
            self._stopping = True

    def set_FromTracker(self, tracker):
        self._tracker = tracker
        return self

    def add(self, mapblock, **kw):
        self.tasks.put((None, mapblock, kw))
        return self

    # see RANdom SAmple Consensus algorithm
    def _merge(self, cur, pre):
        print("[Thread %d] merging block %s and block %s" % (cur, pre))

        pass

    # see RANdom SAmple Consensus algorithm
    def _merge(self, cur, pre, kw):
        print("[Thread %d] merging block %s and block %s" % (self.seq, cur, pre))
        matched = kw.get('matched', None)
        if matched is None:
            raise Exception("Wrong KeyValue '%s' !" % 'matched')

        unmtched_roi = kw.get("unmtched_roi", None)
        if unmtched_roi is None:
            raise Exception("Wrong KeyValue '%s' !" % 'unmtched_roi')

        # constructing ICP solver
        pcloud_src = []
        pcloud_dest = []

        pcloud_src_keys = []
        pcloud_dest_keys = []

        # using landmark centroids
        for landmark, roi in matched:
            p1 = landmark.Centroid()
            p2 = roi.Centroid()

            if p1 is None or p2 is None:
                continue

            pcloud_src.append(p1.data)
            pcloud_dest.append(p2.data)

            # add keys
            p1.key = p1.seq
            p2.key = p2.seq

            # add to pointcloud
            cur.keypointCloud[p1.key] = p1
            pre.keypointCloud[p2.key] = p2

            pcloud_src_keys.append(p1.key)
            pcloud_dest_keys.append(p2.key)

            # retrieve and match points
            sub_src_keys = np.array([k for k, _ in landmark.points.items()])
            sub_dest_keys = np.array([k for k, _ in roi.points.items()])

            pcl_sub_src = np.array([landmark.points[k].data for k in sub_src_keys])
            pcl_sub_dest = np.array([roi.points[k].data for k in sub_dest_keys])

            print("====================================================")
            print("%s pcl_sub_src.shape: " % landmark, pcl_sub_src.shape)
            print("%s pcl_sub_dest.shape: " % roi, pcl_sub_dest.shape)

            if pcl_sub_src.shape[0] == 0 or pcl_sub_dest.shape[0] == 0:
                continue

            if pcl_sub_src.shape[0] <= pcl_sub_dest.shape[0] / 2:
                pass  # continue

            if pcl_sub_src.shape[0] >= pcl_sub_dest.shape[0] * 2:
                pass  # continue

            size = np.min((pcl_sub_src.shape[0], pcl_sub_dest.shape[0]))
            src_mask = np.random.choice(pcl_sub_src.shape[0], size, replace=False)
            print("Sampling %d points from pcl_sub_src" % size)
            pcl_sub_src = pcl_sub_src[src_mask]

            dest_mask = np.random.choice(pcl_sub_src.shape[0], size, replace=False)
            print("Sampling %d points from pcl_sub_dest" % size)
            pcl_sub_dest = pcl_sub_dest[dest_mask]

            # using close form to estimate an initial transformation
            R_i, t_i = self._optimizer.svd_solver(pcl_sub_src, pcl_sub_dest)
            pcl_sub_src0 = self._optimizer.transform(pcl_sub_src, R_i, t_i)

            # KNN
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pcl_sub_dest)
            dist, indices = nbrs.kneighbors(pcl_sub_src0, return_distance=True)

            # extract
            N = len(indices)
            for i in range(N):
                pcloud_src.append(pcl_sub_src[i])
                if i % 100 == 0:
                    print("Adding point %s to src" % pcl_sub_src[i])
                pcloud_dest.append(pcl_sub_dest[indices[i][0]])
                if i % 100 == 0:
                    print("Adding point %s to dest" % pcl_sub_dest[indices[i][0]])

                # add keys
                pcloud_src_keys.append(sub_src_keys[i])
                pcloud_dest_keys.append(sub_dest_keys[indices[i][0]])

        pcloud_src = np.array(pcloud_src)
        pcloud_dest = np.array(pcloud_dest)

        print("pcloud_src.shape: ", pcloud_src.shape)
        print("pcloud_dest.shape: ", pcloud_dest.shape)

        if DEBUG:
            ax = plot_data(pcloud_src[:, 0], pcloud_src[:, 1], pcloud_src[:, 2])
            ax = plot_data(pcloud_dest[:, 0], pcloud_dest[:, 1], pcloud_dest[:, 2], ax=ax, color='r')

            plt.show()
            pass

        estimated_pose, matches = self._optimizer._icp_point_point(pcloud_src, pcloud_dest, None, None,
                                                                   callback=user_plot_wrapper(dest=pcloud_dest))

        #
        R, t = self._optimizer._get_pose(estimated_pose)
        print("got estiamted pose: \nR,\n%s\nt,\n%s\n" % (
            R,
            t
        ))

        matched_points = {}
        unmtched_points = {}
        N = matches.shape[0]
        print("Adding matched points ...")
        for i in range(N):
            key0 = pcloud_src_keys[matches[i, 0]]
            key1 = pcloud_dest_keys[matches[i, 1]]

            matched_points[key1] = (pre.keypointCloud[key1], cur.keypointCloud[key0])
        print("%d matched points." % len(matched_points))

        print("Adding unmatched points ...")
        for k, p in pre.keypointCloud.items():
            q = matched_points.get(k, None)
            if q is None:
                unmtched_points[k] = p
        print("%d unmatched points." % len(unmtched_points))

        # only allow one thread to modify the mapblock
        # @todo : TODO using concurrent container to allow finer concurrent operations
        with cur.modifying_mapblock_locker:
            # set condition
            # wait for localmapping to complete
            print("[Thread %d] Relocalization._merge, waiting for peer modifiers to complete ..." % \
                  (self.seq,))
            cur.modifying_completion_condition.wait_for(lambda: cur.complete_modifying)
            print("[Thread %d] Relocalization._merge, begin to modify the mapblock ..." % \
                  (self.seq,))
            cur.complete_modifying = False

            #
            cur.UpdateMap(R, t)
            cur.Merge(pre, matched, unmtched_roi, matched_points, unmtched_points)
            print("[Thread %d] Relocalization._merge, merge two maps successfully!")

        # notify the main thread that we are ready and update coordinates
        cur.set_complete_updating()

        with self._tracker._state_modifier_lock:
            self._tracker._is_relocalization_mode = False
        return cur

    def Merge(self, kw):
        #     while len(self._mapblocks) > 0:
        #       cur = self._mapblocks.pop()
        #       pre = self._mapblocks.pop()

        #       # do merge

        #       merged = self._merge(cur, pre)

        #       self._mapblocks.append(merged)

        #
        pre = self._mapblocks.pop()
        merged = self._merge(self.cur_map, pre, kw)
        self._mapblocks.append(merged)
        pass

    # override
    def _run_once(self):
        print("[Thread %d] fetching a mapblock..." % \
              (self.seq,))
        _, mapblock, kw = self.tasks.get()
        print("[Thread %d] a new mapblock %s fetched." % \
              (self.seq, mapblock))
        try:
            print("[Thread %d] running relocalization ICV/ICL/BipartiteMatching algorithms upon the new mapblock %s" % \
                  (self.seq, mapblock))

            self.cur_map = mapblock

            if self.reference_map is None:
                self.reference_map = self.cur_map

            if len(self._mapblocks) == 0:
                # with self._maps_lock:
                self._mapblocks.append(mapblock)
                pass
            else:
                # trigger ICP algorithms
                self.Merge(kw)
            pass

        except RuntimeErr as e:
            self.logger.error(e)
        finally:
            self.tasks.task_done()