import os
import sys
try:
  # PY.VERSION <  3.6
  reduce
except:
  # PY.VERSION >= 3.6
  from functools import reduce
import numpy as np

import logging

_logger = logging.getLogger("bundle_adjust")

from svso.lib.sys import add_path
from svso.lib.misc import AtomicCounter

# pwd = os.path.dirname(os.path.realpath(__file__))
# add_path(os.path.join(pwd, '..', 'config'))

# from config import Settings
from svso.config import Settings

# print(sys.path)
# add compiled pybind11 binding lib of g2o for python
class OptimizerConfig(Settings):
  pass

optimizer_config = OptimizerConfig("svso.optimizers.settings")
add_path(optimizer_config.G2OPYLIB_ROOT)

import g2o

# print(dir(g2o))

class Optimizer(g2o.SparseOptimizer):

  def __init__(self):
    super().__init__()

  def vertex_seq_generate(self, entity_name, entity_idx):
    v_idx = self.vertex_seq_generator()
    vertice = self._ivq.get(entity_name, None)
    if vertice is None:
      vertice = {}
      self._ivq[entity_name] = vertice
    vertice[entity_idx] = v_idx
    return v_idx

  def indexOfVertex(self, entity_name, entity_idx):
    vertice = self._ivq.get(entity_name, None)
    if vertice is None:
      raise Exception("No %s has been registered" % entity_name)
    v_idx = vertice.get(entity_idx, None)
    if v_idx is None:
      raise Exception("Could not find vertex for %s %d" % (entity_name, entity_idx))
    return v_idx

  def get_pose(self, entity_name, frame_key):
    v_idx = self.indexOfVertex(entity_name, frame_key)
    pose = self.vertex(v_idx).estimate()
    ##
    pose = pose.inverse()
    ##
    return pose

  def get_point(self, entity_name, point_key):
    v_idx = self.indexOfVertex(entity_name, point_key)
    point = self.vertex(v_idx).estimate()
    return point

  def edge_seq_generate(self, key):
    e_idx = self.edge_seq_generator()
    _KEY_NAME = "VERTEX,VERTEX"
    edges = self._ieq.get(_KEY_NAME, None)
    if edges is None:
      edges = {}
      self._ieq[_KEY_NAME] = edges
    edges[key] = e_idx
    return e_idx

  def EstimateError(self):
    frames = self._map.get_active_frames()
    landmarks = self._map.trackList()
    pointCloud = self._map.trackPointsList()

    rsmes = [None, 0.]
    for i, frame in enumerate(frames):
      cam = frame.camera
      if cam is None:
        continue
      refined_pose = self.get_pose("Frame", frame.seq)
      # @todo : TODO estimate error of poses

    for _, point in pointCloud.items():
      refined_point = self.get_point("MapPoint", point.seq)
      err = point.data- refined_point
      rsmes[1] += np.sum(err ** 2)

    if self.USE_LANDMARKS:
      for _, landmark in landmarks.items():
        centroid = landmark.Centroid()
        refined_point = self.get_point("MapPoint", centroid.seq)
        err = point.data - refined_point
        rsmes[1] += np.sum(err ** 2)
        pass
      pass

    if self.USE_LANDMARKS:
      rsmes[1] /= float(len(pointCloud) + len(landmarks))
    else:
      rsmes[1] /= float(len(pointCloud))
    return rsmes

  def UpdateMap(self):
    frames = self._map.get_active_frames()
    landmarks = self._map.trackList()
    pointCloud = self._map.trackPointsList()

    for i, frame in enumerate(frames):
      cam = frame.camera
      if cam is None:
        continue
      frame.update_pose(self.get_pose("Frame", frame.seq))

    for _, point in pointCloud.items():
      refined_point = self.get_point("MapPoint", point.seq)
      print("refined mappoint %s position:" % point, refined_point)
      point.update(*refined_point)

    if self.USE_LANDMARKS:
      for _, landmark in landmarks.items():
        centroid = landmark.Centroid()
        refined_point = self.get_point("MapPoint", centroid.seq)
        print("refined landmark centroid %s position:" % centroid, refined_point)
        centroid.update(*refined_point)
        pass
      pass


# For the memoment we use g2o for fast development, later we will use our own graph optimizer based
# on SBA algortihm
class BundleAdjustment(Optimizer):

  def __init__(self):
    super().__init__()
    # g2o::BlockSlover_6_3(g2o::BlockSolver_6_3::LinearSolverType*)
    linear_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(linear_solver)
    super().set_algorithm(solver)

    # additional parameters

    #
    self._map = None

    #
    self.vertex_seq_generator = AtomicCounter()
    self.edge_seq_generator = AtomicCounter()

    # Point | Frame | Landmark -> Vertex mapping
    # inverse vertex query
    self._ivq = {}

    # (Vertex, Vetex) -> Edge mapping, a sparse matrix
    # inverse edges query
    self._ieq = {}

    #
    self.USE_LANDMARKS = False

  def set_FromMap(self, map):
    self._map = map
    return self

  # def Init(self):
  #   frames = self._map.get_active_frames()
  #   landmarks = self._map.trackList()
  #   pointCloud = self._map.trackPointsList()
  #
  #   # construct graph
  #   # set key frame as vertices
  #   for i, frame in enumerate(frames):
  #     cam = frame.camera
  #     pose = None
  #     if cam is None:
  #       continue
  #     pose = g2o.SE3Quat(cam.R0, cam.t0.reshape(3, ))
  #     v_idx = self.vertex_seq_generate("Frame", frame.seq)
  #     # only set the first frame as stational piont
  #     # self.add_pose(v_idx, pose, False)#fixed=frame.seq == 1)
  #
  #     # when use ground truth
  #     self.add_pose(v_idx, pose, fixed=frame.seq == 1)
  #
  #   # set array of MapPoint as vertices
  #   for _, point in pointCloud.items():
  #     v = point.data
  #     v_idx = self.vertex_seq_generate("MapPoint", point.seq)
  #     self.add_point(v_idx, v, marginalized=True)
  #
  #     # set edges
  #     observations = point.frames
  #
  #     for frame_key, pixel_pos in observations.items():
  #       frame = self._map.findFrame(frame_key)
  #       cam = frame.camera
  #       if cam is None:
  #         continue
  #       key = (v_idx, self.indexOfVertex("Frame", frame_key))
  #       e_idx = self.edge_seq_generate(key)
  #
  #       # measurement
  #       px = frame.pixels[pixel_pos]
  #
  #       # @todo: TODO compute invSigma for : see ORBSlam implementation for details
  #       invSigma = 1.
  #
  #       if not isinstance(key[1], int):
  #         print("key[1]", key[1])
  #         raise Exception("Wrong value!")
  #       edge = self.add_edge(e_idx, key[0], key[1], px.data,
  #                            information=np.identity(2) * invSigma)
  #
  #       # set camera parameters to compute reprojection error with measurements
  #       cam = frame.camera
  #       device = cam.device
  #
  #       # modify python/types/sba/type_six_dof_expmap.h#L81
  #       #
  #       # Projection using focal_length in x and y directions
  #       # py::class_<EdgeSE3ProjectXYZ, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>>(m, "EdgeSE3ProjectXYZ")
  #       #   .def(py::init<>())
  #       #   .def("compute_error", &EdgeSE3ProjectXYZ::computeError)
  #       #   .def("is_depth_positive", &EdgeSE3ProjectXYZ::isDepthPositive)
  #       #   .def("cam_project", &EdgeSE3ProjectXYZ::cam_project)
  #       # + .def_readwrite("fx", &EdgeSE3ProjectXYZ::fx)
  #       # + .def_readwrite("fy", &EdgeSE3ProjectXYZ::fy)
  #       # + .def_readwrite("cx", &EdgeSE3ProjectXYZ::cx)
  #       # + .def_readwrite("cy", &EdgeSE3ProjectXYZ::cy)
  #       #   ;
  #       #
  #       edge.fx = device.fx
  #       edge.fy = device.fy
  #       edge.cx = device.cx
  #       edge.cy = device.cy
  #
  #       # check our modification is correct: I am not sure whether g2opy runs as expected so we check the result manually.
  #
  #       measurement = edge.cam_project(edge.vertex(1).estimate().map(edge.vertex(0).estimate()))
  #       print("Measurement: %s" % measurement)
  #       print("px: %s" % px.data)
  #       # assert int(measurement[0]) == int(px.x) and int(measurement[1]) == int(px.y)
  #       pass
  #
  #   if self.USE_LANDMARKS:
  #     # treat landmark as stationary points group and compute key points from it
  #     logging.info("Using landmarks for bundle adjustment ...")
  #     # set landmarks as vertices
  #     for _, landmark in landmarks.items():
  #       # representing landmark as top centre point
  #       logging.info("%s points size %d" % (landmark, len(landmark.points)))
  #
  #       # compute bbox
  #       # Note: I just use AABB instead of OBB, because to estimate OBB we need dense points
  #       # bbox = landmark.computeAABB()
  #       # topCentre = bbox.topCentre()
  #       # v = topCentre.data
  #
  #       centroid = landmark.Centroid()
  #       v = centroid.data
  #       v_idx = self.vertex_seq_generate("MapPoint", centroid.seq)
  #
  #       self.add_point(v_idx, v, marginalized=True)
  #
  #       # estimate measurement
  #       # we suppose the positions (centers) of landmarks are stable, which will be used in PnP later
  #       # @todo : TODO
  #
  #       # set edges
  #       observations = landmark.observations
  #
  #       # choose an estimate pose
  #       visited_pxes = {}
  #       for (point_key, frame_seq), pixel_pos in observations.items():
  #         point = pointCloud[point_key]
  #         frame = frames[frame_seq]
  #         cam = frame.camera
  #         if cam is None:
  #           continue
  #
  #         px_pose = point[frame_seq]
  #         px = frame.pixels[pixel_pos]
  #
  #         if visited_pxes.get(frame_seq, None) is None:
  #           visited_pxes = []
  #         visited_pxes[frame_seq].append(px.data)
  #         pass
  #
  #       for frame_seq, projected_pixels in visited_pxes:
  #         reduced = reduce(lambda pre, cur: pre + cur, projected_pixels)
  #         reduced /= len(projected_pixels)
  #         key = (v_idx, self.indexOfVertex("Frame", frame_seq))
  #         e_idx = self.edge_seq_generate(key)
  #
  #         # @todo: TODO compute invSigma for : see ORBSlam implementation for details
  #         # I have little idea how this should be set
  #         invSigma = 1.
  #
  #         edge = self.add_edge(self.vertexPair_edges[key], key[0], key[1], reduced,
  #                              information=np.identity(2) * invSigma)
  #
  #         # set camera parameters to compute reprojection error with measurements
  #         cam = frame.camera
  #         device = cam.device
  #
  #         # add camera parameters to compute reprojection errors
  #         edge.fx = device.fx
  #         edge.fy = device.fy
  #         edge.cx = device.cx
  #         edge.cy = device.cy
  #
  #         pass
  #       pass
  #
  #     logging.info("Number of vertices:", len(self.vertices()))
  #     logging.info("Number of edges:", len(self.edges()))
  #     return self

  def Init(self):
      frames = self._map.get_active_frames()
      landmarks = self._map.trackList()
      pointCloud = self._map.trackPointsList()

      once = False

      # construct graph
      # set key frame as vertices
      for i, frame in enumerate(frames):
        cam = frame.camera
        pose = None
        if cam is None:
          continue
        pose = g2o.SE3Quat(cam.R0, cam.t0.reshape(3, ))
        v_idx = self.vertex_seq_generate("Frame", frame.seq)
        # only set the first frame as stational piont
        # self.add_pose(v_idx, pose, False)#fixed=frame.seq == 1)

        # when use ground truth
        self.add_pose(v_idx, pose, fixed=frame.seq == 1)

        if not once:
          K = cam.K
          focal_length = (K[0, 0] + K[1, 1]) / 2
          pp = (K[0, 2], K[1, 2])
          cam_p = g2o.CameraParameters(focal_length, pp, 0)
          cam_p.set_id(0)
          self.add_parameter(cam_p)
          once = True

      # set array of MapPoint as vertices
      for _, point in pointCloud.items():
        v = point.data  # + np.random.randn(3)
        v_idx = self.vertex_seq_generate("MapPoint", point.seq)
        self.add_point(v_idx, v, marginalized=True)

        # set edges
        try:
          observations = point.frames
        except AttributeError:
          observations = point.observations

        for frame_key, pixel_pos in observations.items():
          frame = self._map.findFrame(frame_key)
          cam = frame.camera
          if cam is None:
            continue
          key = (v_idx, self.indexOfVertex("Frame", frame_key))
          e_idx = self.edge_seq_generate(key)

          # measurement
          px = frame.pixels[pixel_pos]

          # @todo: TODO compute invSigma for : see ORBSlam implementation for details
          invSigma = 1.

          if not isinstance(key[1], int):
            print("key[1]", key[1])
            raise Exception("Wrong value!")
          edge = self.add_edge(e_idx, key[0], key[1], px.data,  # + np.random.randn(2),
                               information=np.identity(2) * invSigma)

          # set camera parameters to compute reprojection error with measurements
          cam = frame.camera
          device = cam.device

          # modify python/types/sba/type_six_dof_expmap.h#L81
          #
          # Projection using focal_length in x and y directions
          # py::class_<EdgeSE3ProjectXYZ, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>>(m, "EdgeSE3ProjectXYZ")
          #   .def(py::init<>())
          #   .def("compute_error", &EdgeSE3ProjectXYZ::computeError)
          #   .def("is_depth_positive", &EdgeSE3ProjectXYZ::isDepthPositive)
          #   .def("cam_project", &EdgeSE3ProjectXYZ::cam_project)
          # + .def_readwrite("fx", &EdgeSE3ProjectXYZ::fx)
          # + .def_readwrite("fy", &EdgeSE3ProjectXYZ::fy)
          # + .def_readwrite("cx", &EdgeSE3ProjectXYZ::cx)
          # + .def_readwrite("cy", &EdgeSE3ProjectXYZ::cy)
          #   ;
          #
          #         edge.fx = device.fx
          #         edge.fy = device.fy
          #         edge.cx = device.cx
          #         edge.cy = device.cy

          # check our modification is correct: I am not sure whether g2opy runs as expected so we check the result manually.

          # used for EdgeSE3ProjectXYZ
          #         measurement = edge.cam_project( edge.vertex(1).estimate().map( edge.vertex(0).estimate() ) )
          # print("Measurement: %s" % measurement)
          # print("px: %s" % px.data)
          # assert int(measurement[0]) == int(px.x) and int(measurement[1]) == int(px.y)
          pass

      if self.USE_LANDMARKS:
        # treat landmark as stationary points group and compute key points from it
        logging.info("Using landmarks for bundle adjustment ...")
        # set landmarks as vertices
        for _, landmark in landmarks.items():
          # representing landmark as top centre point
          logging.info("%s points size %d" % (landmark, len(landmark.points)))

          # compute bbox
          # Note: I just use AABB instead of OBB, because to estimate OBB we need dense points
          # bbox = landmark.computeAABB()
          # topCentre = bbox.topCentre()
          # v = topCentre.data

          centroid = landmark.Centroid()
          v = centroid.data
          v_idx = self.vertex_seq_generate("MapPoint", centroid.seq)

          self.add_point(v_idx, v, marginalized=True)

          # estimate measurement
          # we suppose the positions (centers) of landmarks are stable, which will be used in PnP later
          # @todo : TODO

          # set edges
          observations = landmark.observations

          # choose an estimate pose
          visited_pxes = {}
          for (point_key, frame_seq), pixel_pos in observations.items():
            point = pointCloud[point_key]
            # frame = frames[frame_seq - 1]
            frame = self._map.findFrame(frame_key)
            cam = frame.camera
            if cam is None:
              continue

            px_pose = point[frame_seq]
            px = frame.pixels[pixel_pos]

            if visited_pxes.get(frame_seq, None) is None:
              visited_pxes = []
            visited_pxes[frame_seq].append(px.data)
            pass

          for frame_seq, projected_pixels in visited_pxes:
            reduced = reduce(lambda pre, cur: pre + cur, projected_pixels)
            reduced /= len(projected_pixels)
            key = (v_idx, self.indexOfVertex("Frame", frame_seq))
            e_idx = self.edge_seq_generate(key)

            # @todo: TODO compute invSigma for : see ORBSlam implementation for details
            # I have little idea how this should be set
            invSigma = 1.

            edge = self.add_edge(self.vertexPair_edges[key], key[0], key[1], reduced,
                                 information=np.identity(2) * invSigma)

            # set camera parameters to compute reprojection error with measurements
            cam = frame.camera
            device = cam.device

            # add camera parameters to compute reprojection errors
            #           edge.fx = device.fx
            #           edge.fy = device.fy
            #           edge.cx = device.cx
            #           edge.cy = device.cy

            pass
          pass

      logging.info("Number of vertices:", len(self.vertices()))
      logging.info("Number of edges:", len(self.edges()))
      return self

  def optimize(self, max_iterations=5, verbose=True):
    super().initialize_optimization()
    super().set_verbose(verbose)
    super().optimize(max_iterations)
    return self

  # @todo :TODO
  # pose: g2o.Isometry3d or g2o.SE3Quat
  def add_pose(self, pose_id, pose, fixed=False):
    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_id(pose_id)
    v_se3.set_fixed(fixed)
    ##

    ##
    v_se3.set_estimate(pose.inverse())

    super().add_vertex(v_se3)
    return v_se3

  # point: numpy array with shape=(3,), similar to Eigen::Vector3
  def add_point(self, point_id, point, fixed=False, marginalized=True):
    v_p = g2o.VertexSBAPointXYZ()
    v_p.set_id(point_id)
    v_p.set_estimate(point)
    v_p.set_marginalized(marginalized)
    v_p.set_fixed(fixed)

    super().add_vertex(v_p)
    return v_p

  # @todo : TODO
  def add_edge(self, edge_id, point_id, pose_id,
               measurement,
               information=np.identity(2),
               robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):
    # edge = g2o.EdgeSE3ProjectXYZ()
    edge = g2o.EdgeProjectXYZ2UV()
    edge.set_id(edge_id)
    edge.set_vertex(0, self.vertex(point_id) )
    edge.set_vertex(1, self.vertex(pose_id) )
    edge.set_measurement(measurement)
    edge.set_information(information)
    if robust_kernel is not None:
      edge.set_robust_kernel(robust_kernel)
    edge.set_parameter_id(0,0)
    super().add_edge(edge)
    return edge


## == Used in mapping thread
class LocalBA(BundleAdjustment):

  def __init__(self):
    super().__init__()

    #
    self.frame = None

    # all the points can be viewed by this frame
    self.points = None

    #
    self.measurements = None

    #
    self.mappoints = []

    #
    self.local_key_frames = {}

    #
    self.Thr = 15

    #
    self.adjusted_frames = None

    #
    self.fixed_frames = None

  def set_FromFrame(self, frame):
    self.frame = frame
    return self

  def set_FromPoints(self, points):
    self.points = points
    return self

  def set_FromMeasurements(self, measurements):
    self.measurements = measurements
    return self

  # override
  def Init(self):
    adjusted, fixed = self.get_local_keyframes()
    print("[LocalBA] %d adjusted frames" % len(adjusted))
    print("[LocalBA] %d fixed frames" % len(fixed))
    self.adjusted_frames = adjusted
    self.fixed_frames = fixed

    once = False

    # construct graph
    # set key frame as vertices
    for i, frame in enumerate(adjusted):
      cam = frame.camera
      pose = None
      if cam is None:
        continue
      pose = g2o.SE3Quat(cam.R0, cam.t0.reshape(3, ))
      v_idx = self.vertex_seq_generate("Frame", frame.seq)
      # only set the first frame as stational piont
      # self.add_pose(v_idx, pose, False)#fixed=frame.seq == 1)

      # when use ground truth
      self.add_pose(v_idx, pose, fixed=False)

      if not once:
        K = cam.K
        focal_length = (K[0, 0] + K[1, 1]) / 2
        pp = (K[0, 2], K[1, 2])
        cam_p = g2o.CameraParameters(focal_length, pp, 0)
        cam_p.set_id(0)
        self.add_parameter(cam_p)
        once = True

      pointCloud, Measurement = frame.get_measurements()
      N = len(pointCloud)
      for i in range(N):
        point = pointCloud[i]

        v = point.data  # + np.random.randn(3)
        v_idx = self.vertex_seq_generate("MapPoint", point.seq)
        self.add_point(v_idx, v, marginalized=True)

        # set edge
        cam = frame.camera
        if cam is None:
          continue
        key = (v_idx, self.indexOfVertex("Frame", frame.seq))
        e_idx = self.edge_seq_generate(key)

        # measurement
        px = Measurement[i]

        # @todo: TODO compute invSigma for : see ORBSlam implementation for details
        invSigma = 1.

        if not isinstance(key[1], int):
          print("key[1]", key[1])
          raise Exception("Wrong value!")
        edge = self.add_edge(e_idx, key[0], key[1], px.data,  # + np.random.randn(2),
                             information=np.identity(2) * invSigma)

        self.mappoints.append((point, frame.seq, px))

        # set camera parameters to compute reprojection error with measurements
        cam = frame.camera
        device = cam.device

        # modify python/types/sba/type_six_dof_expmap.h#L81
        #
        # Projection using focal_length in x and y directions
        # py::class_<EdgeSE3ProjectXYZ, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>>(m, "EdgeSE3ProjectXYZ")
        #   .def(py::init<>())
        #   .def("compute_error", &EdgeSE3ProjectXYZ::computeError)
        #   .def("is_depth_positive", &EdgeSE3ProjectXYZ::isDepthPositive)
        #   .def("cam_project", &EdgeSE3ProjectXYZ::cam_project)
        # + .def_readwrite("fx", &EdgeSE3ProjectXYZ::fx)
        # + .def_readwrite("fy", &EdgeSE3ProjectXYZ::fy)
        # + .def_readwrite("cx", &EdgeSE3ProjectXYZ::cx)
        # + .def_readwrite("cy", &EdgeSE3ProjectXYZ::cy)
        #   ;
        #
    #         edge.fx = device.fx
    #         edge.fy = device.fy
    #         edge.cx = device.cx
    #         edge.cy = device.cy

    # ===== FIXED =====
    for i, frame in enumerate(fixed):
      cam = frame.camera
      pose = None
      if cam is None:
        continue
      pose = g2o.SE3Quat(cam.R0, cam.t0.reshape(3, ))
      v_idx = self.vertex_seq_generate("Frame", frame.seq)
      # only set the first frame as stational piont
      # self.add_pose(v_idx, pose, False)#fixed=frame.seq == 1)

      # when use ground truth
      self.add_pose(v_idx, pose, fixed=True)

      if not once:
        K = cam.K
        focal_length = (K[0, 0] + K[1, 1]) / 2
        pp = (K[0, 2], K[1, 2])
        cam_p = g2o.CameraParameters(focal_length, pp, 0)
        cam_p.set_id(0)
        self.add_parameter(cam_p)
        once = True

      pointCloud, Measurement = frame.get_measurements()
      N = len(pointCloud)
      for i in range(N):
        point = pointCloud[i]

        v = point.data  # + np.random.randn(3)
        v_idx = self.vertex_seq_generate("MapPoint", point.seq)
        self.add_point(v_idx, v, marginalized=True)

        # set edge
        cam = frame.camera
        if cam is None:
          continue
        key = (v_idx, self.indexOfVertex("Frame", frame.seq))
        e_idx = self.edge_seq_generate(key)

        # measurement
        px = Measurement[i]

        # @todo: TODO compute invSigma for : see ORBSlam implementation for details
        invSigma = 1.

        if not isinstance(key[1], int):
          print("key[1]", key[1])
          raise Exception("Wrong value!")
        edge = self.add_edge(e_idx, key[0], key[1], px.data,  # + np.random.randn(2),
                             information=np.identity(2) * invSigma)

        self.mappoints.append((point, frame.seq, px))

        # set camera parameters to compute reprojection error with measurements
        cam = frame.camera
        device = cam.device

        # modify python/types/sba/type_six_dof_expmap.h#L81
        #
        # Projection using focal_length in x and y directions
        # py::class_<EdgeSE3ProjectXYZ, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>>(m, "EdgeSE3ProjectXYZ")
        #   .def(py::init<>())
        #   .def("compute_error", &EdgeSE3ProjectXYZ::computeError)
        #   .def("is_depth_positive", &EdgeSE3ProjectXYZ::isDepthPositive)
        #   .def("cam_project", &EdgeSE3ProjectXYZ::cam_project)
        # + .def_readwrite("fx", &EdgeSE3ProjectXYZ::fx)
        # + .def_readwrite("fy", &EdgeSE3ProjectXYZ::fy)
        # + .def_readwrite("cx", &EdgeSE3ProjectXYZ::cx)
        # + .def_readwrite("cy", &EdgeSE3ProjectXYZ::cy)
        #   ;
        #
    #         edge.fx = device.fx
    #         edge.fy = device.fy
    #         edge.cx = device.cx
    #         edge.cy = device.cy
    pass

  def get_local_keyframes(self):
    frames = self._map.get_active_frames()
    pointCloud = self.points
    adjusted = set()
    fixed = set()

    # select key frames
    for _, point in pointCloud.items():
      # observations = point.frames
      # for frame_key, pixel_pos in observations.items(): # this only contains key points records
      for frame in frames:
        frame_key = frame.seq

        camera = frame.camera
        # print("viewing point %s using camera from Frame#%d" % (point, frame.seq))

        cam_pt = camera.viewWorldPoint(point)

        projection = camera.view(cam_pt)
        if projection is None:
          continue

        if self.local_key_frames.get(frame_key, None) is None:
          self.local_key_frames[frame_key] = 0

        self.local_key_frames[frame_key] += 1

    for frame_key, cnt in self.local_key_frames.items():
      if cnt > self.Thr:
        frame = self._map.findFrame(frame_key)
        print("[LocalBA] add %s to adjusted" % frame)
        adjusted.add(frame)

    for frame in frames:
      if frame not in adjusted:
        print("[LocalBA] add %s to fixed" % frame)
        fixed.add(frame)

    return adjusted, fixed

  # @todo : TODO
  def EstimateError(self):
    pass

  # @todo : TODO
  def UpdateMap(self):
    landmarks = self._map.trackList()

    for i, frame in enumerate(self.adjusted_frames):
      cam = frame.camera
      if cam is None:
        continue
      frame.update_pose(self.get_pose("Frame", frame.seq))

    for point, frame_key, px in self.mappoints:
      refined_point = self.get_point("MapPoint", point.seq)
      # print("refined mappoint %s position:" % point, refined_point)
      point.update(*refined_point)

    if self.USE_LANDMARKS:
      for _, landmark in landmarks.items():
        centroid = landmark.Centroid()
        refined_point = self.get_point("MapPoint", centroid.seq)
        # print("refined landmark centroid %s position:" % centroid, refined_point)
        centroid.update(*refined_point)
        pass
      pass
    pass

## ===============================
class PoseOptimization(Optimizer):

  def __init__(self):
    super().__init__()
    # g2o::BlockSlover_6_3(g2o::BlockSolver_6_3::LinearSolverType*)
    linear_solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(linear_solver)
    super().set_algorithm(solver)

    # additional parameters
    terminate = g2o.SparseOptimizerTerminateAction()
    terminate.set_gain_threshold(1e-6)
    super().add_post_iteration_action(terminate)

    #
    self._map = None

    #
    self.frame = None

    #
    self.points = None

    #
    self.vSE3 = None

    #
    self.measurements = None

    #
    self.vertex_seq_generator = AtomicCounter()
    self.edge_seq_generator = AtomicCounter()

    # Point | Frame | Landmark -> Vertex mapping
    # inverse vertex query
    self._ivq = {}

    # (Vertex, Vetex) -> Edge mapping, a sparse matrix
    # inverse edges query
    self._ieq = {}

    #
    self.USE_LANDMARKS = False

    #
    self.edges = []

  def set_FromMap(self, map):
    self._map = map
    return self

  def set_FromFrame(self, frame):
    self.frame = frame
    return self

  def set_FromPoints(self, points):
    self.points = points
    return self

  def set_FromMeasurements(self, measurements):
    self.measurements = measurements
    return self

  # @todo : TODO
  def Init(self):
    pointCloud = self.points
    measurements = self.measurements

    once = False

    # set current frame as vertex to be optimized
    cam = self.frame.camera

    if not once:
      K = cam.K
      focal_length = (K[0, 0] + K[1, 1]) / 2
      pp = (K[0, 2], K[1, 2])
      cam_p = g2o.CameraParameters(focal_length, pp, 0)
      cam_p.set_id(0)
      self.add_parameter(cam_p)
      once = True

    pose = g2o.SE3Quat(cam.R0, cam.t0.reshape(3, ))
    v_idx = self.vertex_seq_generate("Frame", self.frame.seq)
    self.vSE3 = self.add_pose(v_idx, pose, False)

    # add point
    # set array of MapPoint as vertices
    for _, point in pointCloud.items():
      v = point.data
      v_idx = self.vertex_seq_generate("MapPoint", point.seq)
      # We only optimize pose, it is also possible to use g2o::EdgeSE3ProjectXYZOnlyPose
      self.add_point(v_idx, v, marginalized=True, fixed=True)

      # viewed by the frame
      key = (v_idx, self.indexOfVertex("Frame", self.frame.seq))
      e_idx = self.edge_seq_generate(key)

      # measurement
      measurement = measurements[point.seq]
      px = measurement.px2d2

      # @todo: TODO compute invSigma for : see ORBSlam implementation for details
      # I have little idea how this should be set
      invSigma = 1.

      edge = self.add_edge(e_idx, key[0], key[1], px.data,
                           information=np.identity(2) * invSigma)

      #
      device = self.frame.camera.device

      # modify python/types/sba/type_six_dof_expmap.h#L81
      #
      # Projection using focal_length in x and y directions
      # py::class_<EdgeSE3ProjectXYZ, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>>(m, "EdgeSE3ProjectXYZ")
      #   .def(py::init<>())
      #   .def("compute_error", &EdgeSE3ProjectXYZ::computeError)
      #   .def("is_depth_positive", &EdgeSE3ProjectXYZ::isDepthPositive)
      #   .def("cam_project", &EdgeSE3ProjectXYZ::cam_project)
      # + .def_readwrite("fx", &EdgeSE3ProjectXYZ::fx)
      # + .def_readwrite("fy", &EdgeSE3ProjectXYZ::fy)
      # + .def_readwrite("cx", &EdgeSE3ProjectXYZ::cx)
      # + .def_readwrite("cy", &EdgeSE3ProjectXYZ::cy)
      #   ;
      #
    #       edge.fx = device.fx
    #       edge.fy = device.fy
    #       edge.cx = device.cx
    #       edge.cy = device.cy
    pass

  # PoseOptimization converges very quickly ()
  def optimize(self, max_iterations=5, verbose=True, level=None):
    if level is not None:
      super().initialize_optimization(level)
    else:
      super().initialize_optimization()
    super().set_verbose(verbose)
    super().optimize(max_iterations)
    return self

  #@todo : TDOO
  def optimizeWhileFiltering(self):
    MaxIter = 5
    it = 0
    vSE3 = self.vSE3
    cam = self.frame.camera
    outliers = {}
    while it < MaxIter:
      vSE3.set_estimate(g2o.SE3Quat(cam.R0, cam.t0.reshape(3, )))
      self.optimize(level=it)

      # @todo : TODO
      # see ORBSlam PoseOptimization

      pass
    pass

  def add_pose(self, pose_id, pose, fixed=False):
    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_id(pose_id)
    v_se3.set_fixed(fixed)
    v_se3.set_estimate(pose.inverse())

    super().add_vertex(v_se3)
    return v_se3

  # point: numpy array with shape=(3,), similar to Eigen::Vector3
  def add_point(self, point_id, point, fixed=False, marginalized=True):
    v_p = g2o.VertexSBAPointXYZ()
    v_p.set_id(point_id)
    v_p.set_estimate(point)
    v_p.set_marginalized(marginalized)
    v_p.set_fixed(fixed)

    super().add_vertex(v_p)
    return v_p

  # @todo : TODO
  def add_edge(self, edge_id, point_id, pose_id,
               measurement,
               information=np.identity(2),
               robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):
    # edge = g2o.EdgeSE3ProjectXYZOnlyPose()
    # edge = g2o.EdgeSE3ProjectXYZ()
    edge = g2o.EdgeProjectXYZ2UV()
    edge.set_id(edge_id)
    edge.set_vertex(0, self.vertex(point_id))
    edge.set_vertex(1, self.vertex(pose_id))
    edge.set_measurement(measurement)
    edge.set_information(information)
    if robust_kernel is not None:
      edge.set_robust_kernel(robust_kernel)
    edge.set_parameter_id(0, 0)
    super().add_edge(edge)
    return edge

if __name__ == "__main__":
    edge = g2o.EdgeSE3ProjectXYZ()
    # edge = g2o.EdgeSE3ProjectXYZOnlyPose()
    # passed!
    print(edge.fx)