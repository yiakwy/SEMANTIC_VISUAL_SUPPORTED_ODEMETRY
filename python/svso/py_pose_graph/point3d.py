import numpy as np
import threading

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

def det3d(t, n):
    ret = Vec3(0, 0, 0)

    ret.x = n.z * t.y - n.y * t.z
    ret.y = n.z * t.x - n.x * t.z
    ret.z = n.y * t.x - n.x * t.y

    ret.y *= -1.
    return ret

class Vec3:

    def __init__(self, x, y, z):
        self.data = np.array([x, y, z])

    @property
    def x(self):
        return self.data[0]

    @x.setter
    def x(self, val):
        self.data[0] = val;
        return self

    @property
    def y(self):
        return self.data[1]

    @y.setter
    def y(self, val):
        self.data[1] = val
        return self

    @property
    def z(self):
        return self.data[2]

    @z.setter
    def z(self, val):
        self.data[2] = val
        return self

    @property
    def T(self):
        self.data = self.data.T
        return self

    def update(self, x, y, z):
        self.data[0] = x
        self.data[1] = y
        self.data[2] = z
        return self

    def normalized(self):
        n = np.linalg.norm(self.data)
        ret_data = self.data.copy()
        ret_data /= n
        ret = Vec3(*ret_data)
        return ret

    def __add__(self, other):
        assert self.data.shape == other.data.shape
        ret_data = self.data + other.data
        ret = Vec3(*ret_data)
        return ret

    def __sub__(self, other):
        assert self.data.shape == other.data.shape
        ret_data = self.data - other.data
        ret = Vec3(*ret_data)
        return ret

    def __neg__(self):
        ret_data = -self.data
        ret = Vec3(*ret_data)
        return ret

    def __dot__(self, other):
        v = self.data.copy().reshape((3,1))
        ret = v.T.dot(v)
        return ret

    def __cross__(self, other):
        v = det3d(self, other)
        return v

    def __mul__(self, other):
        if self.data.shape == other.data.shape:
            return self.__cross__(other)
        else:
            return self.__dot__(other)


class Identity:
    def __init__(self, seq, name=None, id=None, uuid=None, tok=None):
        self.seq = seq
        self.name = name
        self.id = id
        self.uuid = uuid
        self.tok = tok

    def __str__(self):
        return "Identity#%d" % self.seq

class Point3D(Vec3):

    Seq = AtomicCounter()

    def __init__(self, x, y, z=0):
        super().__init__(x, y, z)
        self.identity = Identity(Point3D.Seq())

    @property
    def seq(self):
        return self.identity.seq

    @seq.setter
    def seq(self, val):
        self.identity.seq = val
        return self

    def __str__(self):
        return "<Point3D %d: %.3f, %.3f, %.3f>" % (
            self.identity.seq,
            self.x,
            self.y,
            self.z
        )

class MapPoint:

    def __init__(self, x=0, y=0, z=0):
        self.identity = Identity(-1)
        self.p = Point3D(x, y, z)
        self._isBad = False

    @property
    def x(self):
        return self.p.x

    @property
    def y(self):
        return self.p.y

    @property
    def z(self):
        return self.p.z

    def setBadFlag(self):
        self._isBad = True
        return self

    def isBad(self):
        return self._isBad

    def __getitem__(self, i):
        return self.p.data[i]

    def __setitem__(self, i, val):
        self.p.data[i] = val
        return self

    def update(self, x, y, z):
        self.p.x = x
        self.p.y = y
        self.p.z = z
        return self

    @property
    def seq(self):
        return self.identity.seq

    @seq.setter
    def set(self, val):
        self.identity.seq = val
        return self

    @property
    def data(self):
        return self.p.data.copy()

    def __add__(self, other):
        if isinstance(other, [int, float]):
            return self.__class__(self.x + other, self.y + other, self.z + other)
        else:
            ret_data = self.data + other.data
            return self.__class__(*ret_data)

    def __sub__(self, other):
        if isinstance(other, [int, float]):
            return self.__class__(self.x - other, self.y - other, self.z - other)
        else:
            ret_data = self.data - other.data
            return self.__class__(*ret_data)


class CameraPoint3D(MapPoint):

    Seq = AtomicCounter()

    def __init__(self, x, y, z):
        super().__init__()
        self.identity = Identity(CameraPoint3D.Seq())

        self.p = Point3D(x, y, z)

        self.triangulated = False

        self.parent = None

        self.px = None

        self.world = None

    def set_world(self, world_point):
        self.world = world_point
        return self

    def set_px(self, px):
        self.px = px
        return self

    def __str__(self):
        return "<CameraPoint3D %d: %.3f, %.3f, %.3f>" % (
            self.identity.seq,
            self.x,
            self.y,
            self.z
        )


class WorldPoint3D(MapPoint):

    Seq = AtomicCounter()

    def __init__(self, x, y, z):
        super().__init__()
        self.identity = Identity(WorldPoint3D.Seq())

        self.p = Point3D(x, y, z)

        # FrameKey = np.uint64
        # PixelPos = np.uint32
        self.observations = {}

        self.parent = None

        self.color = None

    def set_color(self, color):
        self.color = color
        return self

    def associate_with(self, frame, pixel_pos):
        last_pos = self.observations.get(frame.seq, None)
        px = frame.pixels.get(pixel_pos, None)
        if px is None:
            raise Exception("The pixel is not recorded by the frame!")
        if last_pos is None:
            self.observations[frame.seq] = pixel_pos
        else:
            if last_pos != pixel_pos:
                print("pos %d is different from last_pos %d" % (pixel_pos, last_pos))
                print("Pixel(frame=%d, r=%d, c=%d, pixel_pos=%d) mapped points: " % (
                    frame.seq,
                    px.r,
                    px.c,
                    pixel_pos
                ))
                for p in px.cam_pt_array:
                    w = p.world
                    print(w)
                print("\n")
                legacy_px = frame.pixels.get(last_pos)
                print("Pixel(frame=%d, r=%d, c=%d, last_pos=%d) mapped points: " % (
                    frame.seq,
                    legacy_px.r,
                    legacy_px.c,
                    last_pos
                ))
                for p in legacy_px.cam_pt_array:
                    w = p.world
                    print(w)

                raise Exception("The point %s has already been mapped to a different palce" % self)

        return self

    def __str__(self):
        return "<WorldPoint3D %d: %.3f, %.3f, %.3f>" % (
            self.identity.seq,
            self.x,
            self.y,
            self.z
        )

if __name__ == "__main__":
    cam_pt = CameraPoint3D(0, 0, 0)
    p = WorldPoint3D(0, 0, 0)
    cam_pt.set_world(p)
    print(cam_pt)
    print(p)