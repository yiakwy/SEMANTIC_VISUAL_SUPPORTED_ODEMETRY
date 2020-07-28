from evo.core.transformations import quaternion_matrix, rotation_matrix, euler_from_matrix
import numpy as np

import logging
from pysvso.lib.log import LoggerAdaptor

_logger = logging.getLogger("maths.rotation")

import math

def MultiplyAll(*matrice):
    R = np.identity(4)
    for Ri in matrice:
        R = R.dot(Ri)
    return R


class Euler:
    def __init__(self, x, y, z, order="rXYZ"):
        self.order = order

        self.roll = x
        self.yaw = y
        self.pitch = z
        if order == "rXYZ":
            self.Init_FromAirplane(x, y, z)
        else:
            raise Exception("Not Implemented Yet!")

    def update(self, x, y, z):
        self.roll = x
        self.yaw = y
        self.pitch = z
        if self.order == "rXYZ":
            self.Init_FromAirplane(x, y, z)
        return self

    def Init_FromAirplane(self, roll, yaw, pitch):
        Rx = rotation_matrix(roll,  [1, 0, 0])
        Ry = rotation_matrix(yaw,   [0, 1, 0])
        Rz = rotation_matrix(pitch, [0, 0, 1])
        # Euler its self has different way of updating
        self.R = MultiplyAll(Rx, Ry, Rz)
        return self.R

    @staticmethod
    def fromMatrix(R, order="rXYZ"):
        if order == "rXYZ":
            x, y, z = euler_from_matrix(R, "rxyz")
            return Euler(x, y, z)
        else:
            raise Exception("Not Implemented Yet!")

    pass


# author : Lei Wang
# source: utils/math/quaternion.js
# Date: Created at 2018-07-28
#       Updated at 2020-05-08
# ref : https://www.haroldserrano.com/blog/developing-a-math-engine-in-c-implementing-quaternions
#       see http://wikipedia.org/wiki/Quaternion
#       see three.js, src/math/Quaternion.js
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

        # later we will change it to vec3 instance
        self.q = None

        self.UpdateImaginary()
        self.UpdateLength()
        pass

    def UpdateImaginary(self):
        q = self.q
        if q is not None:
            del q
        q = np.array([self.x, self.y, self.z]).reshape((3, 1))
        self.q = q
        return self

    def UpdateLength(self):
        data = np.array([self.w, self.x, self.y, self.z])
        self._length = np.linalg.norm(data)
        return self

    def Matrix4(self):
        data = np.array([self.w, self.x, self.y, self.z])
        return quaternion_matrix(data)

    @staticmethod
    def fromVector(real, imaginary):
        raise Exception("Not Implemented yet!")

    @classmethod
    def fromAxis(cls, axis, angle):
        raise Exception("Not Implemented yet!")

    pass

# helpers resolving the differential of rotation matrice
def dRx(Rx):
    c = Rx[1, 1]
    s = Rx[2, 1]
    return np.array([
        [0., 0., 0.],
        [0., -s, -c],
        [0.,  c, -s]
    ])

def dRy(Ry):
    c = Ry[0, 0]
    s = Ry[0, 2]
    return np.array([
        [-s, 0.,  c],
        [0., 0., 0.],
        [-c, 0., -s]
    ])

def dRz(Rz):
    c = Rz[0, 0]
    s = Rz[1, 0]
    return np.array([
        [-s, -c, 0.],
        [ c, -s, 0.],
        [0., 0., 0.]
    ])
