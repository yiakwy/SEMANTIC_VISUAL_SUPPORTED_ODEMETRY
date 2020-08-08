import sys
import cv2
import numpy as np

from pysvso.py_pose_graph.point3d import Point3D
from pysvso.lib.log import LoggerAdaptor, init_logging
import logging

init_logging()

_logger = logging.getLogger("PCViewer")

import OpenGL.GL as gl
import pangolin

import multiprocessing
from multiprocessing import Process, JoinableQueue, Lock


class Attr(dict):
    pass


# in c++ we could implement by using variant mechanices
class Datum:

    def __init__(self):
        self.attrs = Attr()


class LandmarkDatum(Datum):

    def __init__(self, landmark):
        super().__init__()

        self.seq = landmark.seq
        self.id = None
        self.uuid = landmark.uuid

        # rendering attributes
        self.color = landmark.color
        try:
            self.abox = landmark.computeAABB()
            print("Landmark %s aabb: \n%s" % (landmark, self.abox.toArray()))
        except Exception as e:
            print(e)
            print("No enough points (%d) for Landmark %s " % (len(landmark.points), landmark))
            self.abox = None


class UICom:

    def __init__(self):
        self.ui = None
        self.attrs = Attr()

    @staticmethod
    def Create(ui):
        uicom = UICom()
        uicom.ui = ui
        return uicom


class Object3d:

    def __init__(self):
        # local transformation relative to its parent
        self.mat = None
        self.matInverse = None

        # world matrix relative to the origin
        self.worldMat = None
        self.worldMatInverse = None

        self.isFocused = False

    def UpdateWorldMat(self):
        pass


class GL_Camera(Object3d):

    def __init__(self):
        super().__init__()
        self.dirs = None
        self.eye = np.array([50., 1., 50.])
        self.up = np.array([0., -1., 0.])

        self.isFocused = True
        self.width = 320
        self.height = 240

        # identity matrix
        self.pose = pangolin.OpenGlMatrix()

        self.texture = None

        # matrice stack
        self.modelMat = None
        self.modelMatInverse = None
        self.viewMat = None
        self.viewMatInverse = None
        self.projectionMat = None
        self.projectionMatInverse = None

        self._rendered_cam = None
        self.view_ports = []

    def Init(self):
        pass

    def set_position(self, pos):
        self.eye = pos
        return self

    # @todo : TODO
    def create_viewport(self, w, h):
        logging.info("Creating main viewport")
        main_viewport = pangolin.CreateDisplay()
        main_viewport.SetBounds(0.0, 1.0, 0.0, 1.0, -640. / 480.)
        main_viewport.SetHandler(pangolin.Handler3D(self._rendered_cam))

        self.view_ports.append(main_viewport)

        logging.info("Creating image viewport")
        image_viewport = pangolin.Display('image')
        image_viewport.SetBounds(0, h / 480., 0., w / 640., 640. / 480.)
        image_viewport.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        self.view_ports.append(image_viewport)

        # set up texture
        logging.info("Creating texture instance to render image ... ")
        self.texture = pangolin.GlTexture(self.width, self.height, gl.GL_RGB, False, 0, gl.GL_UNSIGNED_BYTE)

        return self

    # @todo : TODO
    def UpdateProjectionMat(self):
        raise Exception("Not implemented Yet!")

    # @todo : TODO
    def LookAt(self, dest):
        self.viewMat = pangolin.ModelViewLookAt(self.eye[0], self.eye[1], self.eye[2],
                                                dest[0], dest[1], dest[2],
                                                pangolin.AxisDirection.AxisY)  # self.up[0], self.up[1], self.up[2])
        self._rendered_cam = pangolin.OpenGlRenderState(self.projectionMat, self.viewMat)

        return self.viewMat

    def Activate(self):
        self.view_ports[0].Activate(self._rendered_cam)
        return self

    def RenderImg(self, img):
        img = cv2.resize(img, (self.width, self.height))
        viewport = self.view_ports[1]
        self.texture.Upload(img, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        viewport.Activate()
        gl.glColor3f(1., 1., 1.)
        self.texture.RenderToViewport()

        return self

    # @todo : TODO
    def UpdateModelMat(self):
        pass


class PerspectiveCamera(GL_Camera):

    def __init__(self, fov, aspect, near, far):
        super().__init__()
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        # update MVP matrix stack
        self.UpdateProjectionMat()

    def UpdateProjectionMat(self):
        self.projectionMat = pangolin.ProjectionMatrix(1024, 768, 420, 420,
                                                       320, 240, self.near, self.far)

        return self.projectionMat


class FirstPersonController:

    def __init__(self, cam):
        self._cam = cam

    def Update(self, pose):
        rendered_cam = self._cam._rendered_cam
        # rendered_cam.Follow(pose, True)
        self._cam.Activate()
        return self


def drawABox(aabb, color=None):
    gl.glLineWidth(3)
    if color is None:
        gl.glColor3f(1., 0., 1.)
    else:
        gl.glColor3f(color[0], color[1], color[2])

    x1, y1, z1 = aabb[0]
    x2, y2, z2 = aabb[1]

    w = x2 - x1
    h = y2 - y1
    z = z2 - z1

    cx = (x1 + x2) / 2.
    cy = (y1 + y2) / 2.
    cz = (z1 + z2) / 2.

    pose = np.identity(4)
    pose[:3, 3] = np.array([cx, cy, cz])
    size = [w, h, z]

    pangolin.DrawBoxes(np.array([pose]), np.array([size]))

    pass


# multli threads renderer, see another cross platform gl example from https://github.com/yiakwy/Onsite-Blackboard-Code-Interview/blob/master/apollo/cpp/src/modules/gl/viewer.hpp
# reference:
class PCViewer:

    logger = LoggerAdaptor("PCViewer", _logger)

    # these mocked classes will be improved later
    class Pixel2D:
        def __init__(self):
            self.x = None
            self.y = None

    class Window:
        def __init__(self):
            self.width = None
            self.height = None
            self.aspect = None
            self.fd = None
            self.pos = None
            self.name = None
            self.focused = False

    class Mouse:
        def __init__(self):
            self.press_x = None
            self.press_y = None
            self.state = None

    def __init__(self):
        #
        self._map = None

        #
        self._tracker = None

        # attributes queue
        # self.attributes = JoinableQueue()
        self.manager = multiprocessing.Manager()
        self.attributes = self.manager.Queue()

        #
        self.is_3dmode = True;
        self.main_window = self.Window()
        self.main_window.name = "PCViewer"
        self.main_window.width = 1024
        self.main_window.height = 768

        # gl camera to update MVP matrice stack, see opengl 2 and opengl SE for details of implementation
        self.camera = None

        self.control = None

        self.layout_root = UICom()

        # implement event loop interface
        self._state_locker = Lock()
        # close event
        self._stopped = False

        # stop event
        self._stopping = False

        # to obtain true parallel compute ability
        # async gl renderer
        self._gl_renderer_executor = Process(target=self.glMainLoop)
        self._gl_renderer_executor.daemon = True
        pass

    def Start(self):
        self._gl_renderer_executor.start()
        pass

    def set_FromTracker(self, tracker):
        self._tracker = tracker
        return self

    def set_FromMap(self, map):
        self._map = map
        return self

    def set_stopping(self):
        pass

    def isStopping(self):
        return self._stopping

    def Init(self):
        self.InitWindow()
        self.Clear()
        if self.is_3dmode:
            # setup 3D perspective camera
            self.logger.info("Setting up a 3d camera")
            self.Setup3DCamera()
            pass
        else:
            # setup 2D Camera
            raise Exception("2D Othogonal Camera is not implemented yet!")

        self.logger.info("Creating viewp ports ...")
        self.camera.create_viewport(self.camera.width, self.camera.height)

        self.InitControl()
        self.SetLights()

        # Register polling events
        self.InitPangolin()

        # self.Start()
        pass

    def InitWindow(self):
        # set display mode

        # set windows size and position

        # create window
        #     self.main_window.fd = pangolin.CreateWindowAndBind(
        #       self.main_window.name,
        #       self.main_window.width,
        #       self.main_window.height)

        # self.logger.info("Main Window file descriptor (fd) : %s" % self.main_window.fd)

        self.SetDisplayMode()
        self.SetWindowLayout()
        return self

    def SetWindowLayout(self):
        panel = UICom.Create(pangolin.CreatePanel("menu"))
        self.layout_root.attrs["menu"] = panel

        # @todo : TODO add other UI control components to construct UI components tree

        return self

    def SetDisplayMode(self):
        if self.is_3dmode:
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        else:
            raise Exception("Not Implemented Yet!")
        return self

    def Clear(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        return self

    def Setup3DCamera(self):
        self.camera = PerspectiveCamera(None, None, 0.2, 200)
        self.camera.set_position(np.array([-2., 2, -2.]))
        self.camera.LookAt(np.array([0, 0, 0]))
        return self

    def InitControl(self):
        # init FirstPersonController
        self.control = FirstPersonController(self.camera)
        return self

    # @todo : TODO
    def SetLights(self):
        pass

    def InitPangolin(self):
        pass

    # see tum rgdb_benchmark toolkit
    def drawDepthImage(self, frame):
        # dest = os.path.join(TUM_DATA_DIR, frame.depth_img)
        img = frame.img
        # depth_img = cv2.imread(dest, 0)
        depth_img = frame.depth_img

        H, W = img.shape[0:2]
        SCALING_FACTOR = 5  # 5000.0
        R_PRECISION = 1e-4

        datum = Datum()
        datum.attrs['points-redundant'] = []
        datum.attrs['colors-redundant'] = []

        K = frame.camera.K

        def check_eq(retrived, right):
            left = retrived.data
            dist = np.linalg.norm(left - right)
            if dist < R_PRECISION:
                return True
            else:
                return False

        for y in range(H):
            for x in range(W):
                # set RGB color for rendering
                color = frame.img[int(y), int(x)]
                Z = depth_img[int(y), int(x)]
                Z /= SCALING_FACTOR
                if Z == 0: continue
                X = (x - K[0, 2]) * Z / K[0, 0]
                Y = (y - K[1, 2]) * Z / K[1, 1]

                # get pixel
                idx = int(y * W + x)
                px = frame.pixels.get(idx, None)
                p = None

                if True:  # px is None:
                    p = Point3D(X, Y, Z)
                else:
                    p_vec = np.array([X, Y, Z])
                    dist = None
                    best = None

                    for cam_pt in px.cam_pt_array:
                        w = cam_pt
                        if w is None:
                            continue

                        if best is None:
                            best = w
                            dist = np.linalg.norm(best.data - p_vec)
                        else:
                            dist1 = np.linalg.norm(w.data - p_vec)

                            if dist1 < dist:
                                best = w
                                dist = dist1

                    if best is None:
                        p = Point3D(X, Y, Z)
                    else:
                        p = best
                        print("[PCViewer.DrawDepthImage] Updating %s to (%f, %f, %f) " % (
                          p,
                          X,
                          Y,
                          Z
                        ))
                        p.x = X
                        p.y = Y
                        p.z = Z

                # p.set_color(color)

                # camera to world
                v = p.data.reshape((3, 1))
                # Twc
                v = frame.R0.dot(v) + frame.t0
                v = v.reshape((3,))

                p.update(*v)

                datum.attrs['points-redundant'].append(p)
                datum.attrs['colors-redundant'].append(color)

        datum.attrs['pose'] = self._tracker.cur.get_pose_mat()
        self.logger.info("[Main Thread] put datum to queue")
        self.attributes.put(datum)
        pass

    # @todo : TODO update
    def Update(self, cur=None, last=None, flow_mask=None, active_frames=None):
        datum = Datum()

        cur = cur or self._tracker.cur
        last_frame = last or self._tracker.last_frame
        if flow_mask is None:
            flow_mask = self._tracker.flow_mask

        if last_frame and hasattr(last_frame, "rendered_img") and last_frame.rendered_img is not None:
            out_frame = cv2.add(last_frame.rendered_img, flow_mask)
            datum.attrs['img'] = out_frame
            print("put Frame#%d rendered image to datum" % last_frame.seq)
        else:
            datum.attrs['img'] = cur.img
            print("put Frame#%d source image to datum" % cur.seq)

        datum.attrs['pose'] = cur.get_pose_mat()
        print("put Frame#%d's pose\n%s\n to datum" % (cur.seq, datum.attrs['pose']))

        # add curent structure
        datum.attrs['points'] = []
        pointCloud = cur.get_points()
        frame = cur
        for point in pointCloud:
            try:
                pos = point.frames.get(frame.seq, None)
            except AttributeError:
                pos = point.observations.get(frame.seq, None)
            # g2o module might change the ground truth to strange values!
            if pos is None:
                # raise Exception("Wrong Value!")
                print("Wrong Value!")
                continue

            px = frame.pixels[pos]
            x, y = px.data

            # if frame.depth_img is not None:
            # depth_img = frame.depth_img
            #
            # SCALING_FACTOR = 5
            #
            # K = frame.camera.K
            #
            # Z = depth_img[int(y), int(x)]
            # Z /= SCALING_FACTOR
            # if Z == 0:
            #   point.setBadFlag()
            #   continue
            # X = (x - K[0, 2]) * Z / K[0, 0]
            # Y = (y - K[1, 2]) * Z / K[1, 1]
            #
            # point.update(X, Y, Z)
            #
            # # Twc
            # v = point.data.reshape((3,1))
            # v = frame.R0.dot(v) + frame.t0
            # v = v.reshape((3,))
            #
            # #           print("[PCViewer.Update] Updating from %s to (%f, %f, %f)" % (
            # #             point,
            # #             v[0],
            # #             v[1],
            # #             v[2]
            # #           ))
            # point.update(*v)

            if px.roi.label == "chair":
                pass  # print("chair point: ", point)
            if px.roi.label == "book":
                pass  # print("book point: ", point)

            # check if computed projection of that point is inside bbox
            cam_pt = frame.camera.viewWorldPoint(point)
            projection = frame.camera.view(cam_pt)
            if not projection or not px.roi.isIn(projection):
                # remove the point (mighted be optimized out) outside of roi
                y1, x1, y2, x2 = px.roi.bbox
                print("[PCViewer.Update] %s is out of %s's bbox (%f, %f, %f, %f)" % (
                    projection,
                    px.roi,
                    x1, y1, x2, y2
                ))
                px.roi.findParent().remove(point)
                continue

            #       print("put KeyPoint %s to datum" % point)
            datum.attrs['points'].append(point)
        print("put Frame#%d's structure(%d points) to datum" % (cur.seq, len(pointCloud)))

        # add lines
        datum.attrs['lines'] = []
        frames = active_frames or self._map.get_active_frames()
        l = len(frames)
        for i in range(1, l):
            datum.attrs['lines'].append((frames[i - 1].t0.reshape(3, ), frames[i].t0.reshape(3, )))

        # add landmarks
        datum.attrs['landmarks'] = []
        landmarks = cur.get_landmarks()
        for landmark in landmarks:
            # landmark can not be pickled, see docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
            # datum.attrs['landmarks'].append(landmark)
            picklable = LandmarkDatum(landmark)
            if landmark.label not in ("keyboard", "book", "chair"):
                pass
            if landmark.label in ("table", "dining table",):
                continue
            if picklable.abox is not None:
                print("Putting landmark <Landmark#%d, Label %s> to datum, aabox: \n%s" % (
                    landmark.seq,
                    landmark.label,
                    picklable.abox.toArray()
                ))
                datum.attrs['landmarks'].append(picklable)
            pass
        print("put Frame#%d's structure(%d landmarks) to datum" % (cur.seq, len(datum.attrs['landmarks'])))

        self.logger.info("[Main Thread] put datum to queue")
        self.attributes.put(datum)

        pass

    def setStop(self):
        with self._state_locker:
            self._stopping = True

    # @todo : TODO stop
    def Stop(self):
        #
        # self.attributes.join()
        self.attributes.close()

        print("Set glMainLoop stopping")
        self.setStop()

        #
        self._gl_renderer_executor.join()

        #

        pass

    # @todo : TODO
    def Render(self):
        pass

    # @todo : TODO
    def glMainLoop(self):
        self.logger.info("Running gl viewer ...")

        flag = False

        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        panel = pangolin.CreatePanel('menu')
        panel.SetBounds(1.0, 1.0, 0.0, 100 / 640.)

        # Does not work with initialization of window
        # self.Init()

        camera_pose = pangolin.OpenGlMatrix()

        # create scene graph
        #     scene = pangolin.Renderable()
        #     # x : R
        #     # y : G
        #     # z : B
        #     scene.Add(pangolin.Axis())

        #
        rendered_cam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.01, 2000),
            pangolin.ModelViewLookAt(1, 0, 1, 0, 0, 0, 0, 0, 1)
        )
        handler = pangolin.Handler3D(rendered_cam)
        #     handler = pangolin.SceneHandler(scene, rendered_cam)

        # add drawing callback

        #
        viewport = pangolin.CreateDisplay()
        viewport.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
        viewport.SetHandler(handler)

        image_viewport = pangolin.Display('image')
        w = 160
        h = 120
        image_viewport.SetBounds(0, h / 480., 0.0, w / 640., 640. / 480.)
        image_viewport.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        texture = pangolin.GlTexture(w, h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        # geometries
        self.lines = []
        self.points = {}
        self.colors = {}
        self.poses = []

        self.redundant_points = []
        self.redundant_colors = []

        self.landmarks = {}

        #
        img = np.ones((h, w, 3), 'uint8')

        while not pangolin.ShouldQuit():

            datum = None
            if not self.attributes.empty():
                self.logger.info("[GL Process] attributes is not empty, fetching datum ...")
                datum = self.attributes.get()
                self.logger.info("[GL Process] get a datum from the main thread of the parent process")

            if datum is not None:
                # dispatch instructions
                if datum.attrs.get('pose', None) is not None:
                    pose = datum.attrs['pose']
                    # self.camera.pose.m = pose
                    # set Twc
                    camera_pose.m = pose  # np.linalg.inv(pose)
                    # camera_pose.m = pose
                    rendered_cam.Follow(camera_pose, True)
                    self.logger.info("[GL Process] update camera pose matrix got from datum: \n%s" % pose)
                    pass
                pass

            # self.Clear()
            # self.control.Update(self.camera.pose)

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            viewport.Activate(rendered_cam)
            #       scene.Render()

            # Just for test
            # pangolin.glDrawColouredCube(0.1)

            # render graph
            if datum is not None:
                # dispatch data

                # update graph
                if datum.attrs.get('lines', None) is not None:
                    lines = datum.attrs['lines']
                    self.lines.extend(lines)
                    self.logger.info("[GL Process] drawing %d lines ..." % len(lines))

                if datum.attrs.get('pose', None) is not None:
                    pose = datum.attrs['pose']
                    self.poses.append(pose)
                    self.logger.info("[GL Process] drawing a camera with new pose matrix ...")
                    pass

                # update image
                if datum.attrs.get('img', None) is not None:
                    img = datum.attrs['img']
                    # see pangolin issue #180
                    # change cv BGR channels to norm one
                    # img = img[::-1, : ,::-1].astype(np.uint8)
                    img = img.astype(np.uint8)
                    img = cv2.resize(img, (w, h))
                    self.logger.info("[GL Process] drawing image to image viewport ...")

                # show mappoints
                if datum.attrs.get('points', None) is not None:
                    points = datum.attrs['points']
                    if len(points) > 0:
                        # self.points.extend(points)
                        for point in points:
                            self.points[point.seq] = point

                        # colors = np.array([p.color if p.color is not None else np.array([1.0, 0.0, 0.0]) for p in points]).astype(np.float64)
                        colors = [p.color / 255. if p.color is not None else np.array([1.0, 1.0, 0.0]) for p in points]
                        # print("colors: \n%s" % np.array(colors))

                        # colors = [ [1., 1., 0.] for p in points]

                        # self.colors.extend(colors)
                        for i, color in enumerate(colors):
                            point = points[i]
                            self.colors[point.seq] = color

                        self.logger.info("[GL Process] drawing %d points" % len(points))
                        # print("new mappoints: \n%s" % np.array([ p.data for p in points]).astype(np.float64))
                        # print("new colors (default): \n%s" % np.array(colors))
                    else:
                        self.logger.info("[GL Process] no points to be drawn.")

                # redundant points
                if datum.attrs.get('points-redundant', None) is not None:
                    points = datum.attrs['points-redundant']
                    colors = datum.attrs['colors-redundant']

                    for i, p in enumerate(points):
                        self.redundant_points.append(p)
                        self.redundant_colors.append(colors[i] / 255.)

                # show landmarks
                if datum.attrs.get('landmarks', None) is not None:
                    landmarks = datum.attrs['landmarks']
                    for landmark in landmarks:
                        self.landmarks[landmark.seq] = landmark
                    self.logger.info("[GL Process] drawing %d landmarks" % len(landmarks))

                self.attributes.task_done()
                self.logger.info("[GL Process] datum has been processed.")

            ############
            # draw graph
            ############
            line_geometries = np.array([
                [*line[0], *line[1]] for line in self.lines
            ])
            if len(line_geometries) > 0:
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 1.0, 0.0)
                pangolin.DrawLines(line_geometries, 3)

            # pose = self.camera.pose
            pose = camera_pose

            # GL 2.0 API
            gl.glPointSize(4)
            gl.glColor3f(1.0, 0.0, 0.0)

            gl.glBegin(gl.GL_POINTS)
            gl.glVertex3d(pose[0, 1], pose[1, 3], pose[2, 3])
            gl.glEnd()

            ############
            # draw poses
            ############
            if len(self.poses) > 0:
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 0.0, 1.0)

                # poses: numpy.ndarray[float64], w: float=1.0, h_ratio: float=0.75, z_ratio: float=0.6
                pangolin.DrawCameras(np.array(self.poses))

                gl.glPointSize(4)
                gl.glColor3f(0.0, 0.0, 1.0)

                gl.glBegin(gl.GL_POINTS)
                for pose in self.poses:
                    gl.glVertex3d(pose[0, 1], pose[1, 3], pose[2, 3])
                gl.glEnd()

            ################
            # draw mappoints
            ################
            if len(self.points) > 0:
                # points_geometries = np.array([p.data for p in self.points]).astype(np.float64)
                points_geometries = []
                colors = []
                for point_key, p in self.points.items():
                    points_geometries.append(p.data)
                    colors.append(self.colors[point_key])

                points_geometries = np.array(points_geometries).astype(np.float64)
                colors = np.array(colors).astype(np.float64)

                gl.glPointSize(6)
                # pangolin.DrawPoints(points_geometries, np.array(self.colors).astype(np.float64) )
                pangolin.DrawPoints(points_geometries, colors)

                # gl.glPointSize(4)
                # gl.glColor3f(1.0, 0.0, 0.0)
                #
                # gl.glBegin(gl.GL_POINTS)
                # for point in points_geometries:
                #   gl.glVertex3d(point[0], point[1], point[2])
                # gl.glEnd()

            ####################
            # redundant points #
            ####################
            if len(self.redundant_points) > 0:
                points_geometries = []
                colors = []
                for i, p in enumerate(self.redundant_points):
                    points_geometries.append(p.data)
                    colors.append(self.redundant_colors[i])

                points_geometries = np.array(points_geometries).astype(np.float64)
                colors = np.array(colors).astype(np.float64)

                gl.glPointSize(3)
                pangolin.DrawPoints(points_geometries, colors)

            ################
            # draw landmarks
            ################
            for key, landmarkDatum in self.landmarks.items():
                # abox = landmark.computeAABB()
                # drawABox(abox.toArray())

                drawABox(landmarkDatum.abox.toArray(), color=landmarkDatum.color)
                pass

            #############
            # draw images
            #############
            # self.camera.RenderImg(img)
            texture.Upload(img, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            image_viewport.Activate()
            gl.glColor3f(1.0, 1.0, 1.0)
            texture.RenderToViewport()

            pangolin.FinishFrame()

        print("gl program loop stopped")
        with self._state_locker:
            self._stopped = True