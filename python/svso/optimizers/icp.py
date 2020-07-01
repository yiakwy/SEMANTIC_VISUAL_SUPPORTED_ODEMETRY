import numpy as np
from enum import Enum
from svso.lib.maths.rotation import Euler, Quaternion, rotation_matrix, dRx, dRy, dRz

from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
# when point cloud is parse
from sklearn.neighbors import NearestNeighbors

# see FLANN manual https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf
# remember to run 2to3 upon root of the source when you complete downloading the codes!
from pyflann import *
import numpy as np

# used to build computation graph with
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_graphics as tfg

rotation_matrix_3d = tfg.geometry.transformation.rotation_matrix_3d

class ICP:
    class Algorithm(Enum):
        POINT_POINT = 1
        POINT_PLANE = 2  # G-ICP

    def __init__(self, algorithm=None):
        self.algorithm = algorithm or ICP.Algorithm.POINT_POINT

        #
        self.euler = Euler(0, 0, 0)

        #
        self.R = self.euler.R
        self.t = np.zeros((3, 1))

        #
        self.max_iterations = 3

        #
        self.USE_GRADIENT = False

        #
        self.HAND_MADE_OPT = True

    def transform(self, pcloud_src, R, t):
        N = pcloud_src.shape[0]
        matl = pcloud_src.T
        matl = R.dot(matl) + t
        return matl.T

    # @todo : TODO estimate transformation [R|t] between two points cloud
    def costImpl(self, x, pcloud_src, pcloud_dest):
        # print("x.shape: ", x.shape)
        N = pcloud_src.shape[0]
        assert (pcloud_src.shape == pcloud_dest.shape)

        R, t = self._get_pose(x)

        matl = pcloud_src.T  # 3 * N
        matl = R.dot(matl) + t

        dx = matl.T - pcloud_dest

        #     N = dx.shape[0]
        #     _sum = 0.
        #     for i in range(N):
        #       v = dx[i].reshape((3,1))
        #       _sum += v.T.dot(v)[0,0]
        #       pass

        _sum = np.sum(np.diag(np.dot(dx, dx.T)))
        _sum /= N
        return _sum

    # @todo : TODO
    # @state : DEPREACTED see "jac_mat" and "_optimize"
    # reference: g2o/types/icp/types/types_icp.cpp
    # I have tried autograd by HIPS from Harvard University. However, it doesn't work.
    # Supprisingly, tensorflow computes a reasonable gradient.
    def jac_flow(self, x, pcloud_src, pcloud_dest):
        """
        Author : LEI WANG (yiakwy@gmail.com)
        Date : Jun 2, 2020

        p* = R*p + t = [R|t]*p

        partial dp* / partial dt.T ~ I (partial d P* / dt1 = 0 + [1, 0, 0].T)
                                       (parital d P* / dt2 = 0 + [0, 1, 0].T)
                                       (partial d P* / dt3 = 0 + [0, 0, 1].T)

        In matrix form:
        J ~ 3*6 matrix

        """
        # using computing graph
        N = pcloud_src.shape[0]
        assert (pcloud_src.shape == pcloud_dest.shape)

        R0, t0 = self._get_pose(x)

        rotation_matrix_3d = tfg.geometry.tranformation.rotation_matrix_3d

        # tf variables
        pcloud_src = tf.Variable(pcloud_src)
        pcloud_dest = tf.Variable(pcloud_dest)

        # R = tf.constant(R0, dtype=tf.float64)
        angle = tf.constant(np.array([x[0], x[1], x[2]]))
        t = tf.constant(t0, dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(angle)
            # tape.watch(R)
            tape.watch(t)

            _T = tf.transpose

            R = rotation_matrix_3d.from_euler(angle)

            matl = _T(pcloud_src)
            matl = _T(tf.matmul(R, matl) + t)

            dx = matl - pcloud_dest

            _sum = tf.reduce_sum(tf.linalg.diag_part(tf.matmul(dx, _T(dx))))
            _sum /= N

        dTheta = tape.gradient(_sum, angle)
        # dR = tape.gradient(_sum, R)
        dt = tape.gradient(_sum, t)

        #     R1 = R0 + dR * step
        #     t1 = t0 + dt * step

        #     euler = Euler.fromMatrix(R1)

        #     grad = np.array([
        #       euler.roll,
        #       euler.yaw,
        #       euler.pitch,
        #       t1[0],
        #       t1[1],
        #       t1[2]
        #     ]) - estimated_pose

        grad = np.array([
            dTheta[0],
            dTheta[1],
            dTheta[2],
            dt[0],
            dt[1],
            dt[2]
        ])

        del tape

        return grad
        pass

    # @todo : TODO
    def hess(self, x, pcloud_src, pcloud_dest):
        pass

    # @todo : TODO
    def check_gradient(self, x, pcloud_src, pcloud_dest):
        pass

    # @todo : TODO
    def jac_se3(self, x, pcloud_src, pcloud_dest):
        # using computing graph
        N = pcloud_src.shape[0]
        assert (pcloud_src.shape == pcloud_dest.shape)

        R0, t0 = self._get_pose(x)

        matl = pcloud_src.T  # 3 * N
        matl = R0.dot(matl) + t0

        dx = matl.T - pcloud_dest

        # ====

        Rx = rotation_matrix(x[3], [1, 0, 0])[:3, :3]
        Ry = rotation_matrix(x[4], [0, 1, 0])[:3, :3]
        Rz = rotation_matrix(x[5], [0, 0, 1])[:3, :3]

        dRx_ = dRx(Rx)
        dRy_ = dRy(Ry)
        dRz_ = dRz(Rz)

        # hard coded
        J = np.zeros((3, 6, N))

        J[0:3, 0:3, :] = np.eye(3)[:, :, np.newaxis]
        temp = dRx_.dot(Ry.dot(Rz.dot(pcloud_src.T)))
        print("temp shape:", temp.shape)
        J[0:3, 3, :] = temp
        J[0:3, 4, :] = Rx.dot(dRy_.dot(Rz.dot(pcloud_src.T)))
        J[0:3, 5, :] = Rx.dot(Ry.dot(dRz_.dot(pcloud_src.T)))

        return dx, J

    # implemented and refactored based on the algorithm elabrated in the following talk
    # ref : dis.uniroma1.it/~labrococo/tutorial_icra_2016/icra16_slam_tutorial_grisetti.pdf
    # note : does not work as expected for the moment, Jun 16, 2020
    def jac_se3_manifold(self, x, pcloud_src, pcloud_dest):
        # using computing graph
        N = pcloud_src.shape[0]
        assert (pcloud_src.shape == pcloud_dest.shape)

        R0, t0 = self._get_pose(x)

        matl = pcloud_src.T  # 3 * N
        matl = R0.dot(matl) + t0

        dx = matl.T - pcloud_dest

        # ====

        # hard coded
        J = np.zeros((3, 6, N))

        J[0:3, 0:3, :] = np.eye(3)[:, :, np.newaxis]

        def _Skew(pred):
            skew = np.zeros((3, 3, N))
            for i in range(N):
                v = pred[:, i].reshape((3,))
                skew[:, :, i] = np.array([
                    [0., -v[2], v[1]],
                    [v[2], 0., -v[0]],
                    [-v[1], v[0], 0.]
                ])
            return skew

        J[0:3, 3:6, :] = _Skew(matl)
        return dx, J

    # implemented and refactored based on the algorithm elabrated in the following talk
    # ref : dis.uniroma1.it/~labrococo/tutorial_icra_2016/icra16_slam_tutorial_grisetti.pdf
    # note : does not work as expected for the moment, Jun 16, 2020
    def _minimize(self, pcloud_src, pcloud_dest, estimated_pose, max_iter=10):
        # find optimal H such that incremental function 0 = H * dx holds
        x = estimated_pose.copy()
        chi_stats = np.zeros((max_iter,))
        for k in range(max_iter):
            H = np.zeros((6, 6))
            b = np.zeros((6, 1))
            chi = 0.

            #       dx, J = self.jac_se3(x, pcloud_src, pcloud_dest)
            # manifold method
            dx, J = self.jac_se3_manifold(x, pcloud_src, pcloud_dest)

            N = pcloud_src.shape[0]
            for i in range(N):
                dxi = dx[i, :].reshape((3, 1))
                err = dxi.T.dot(dxi)
                print("err_i : %f" % err)
                if err > 1.:
                    continue
                # print("dxi shape:", dxi)
                Ji = J[:, :, i]
                # print("Ji shape:", Ji)
                H += Ji.T.dot(Ji)
                temp = Ji.T.dot(dxi)
                # print("temp shape:", temp)
                b += temp
                chi += err

            chi /= N
            chi_stats[k] = chi
            # might produce huge error here
            delta = -np.linalg.solve(H, b).reshape((6,))
            #       x += delta

            # manifold method
            R1, t1 = self._get_pose(delta)
            R0, t0 = self._get_pose(x)
            R = R1.dot(R0)
            t = R1.dot(t0) + t1

            euler = Euler.fromMatrix(R)
            x = np.array([
                euler.roll,
                euler.yaw,
                euler.pitch,
                t[0][0],
                t[1][0],
                t[2][0]])

            print("Iter %d, chi %f, delta %s, x %s" % (k, chi, delta, x))

        return x, chi_stats[-1]
        pass

    # see github.com/gramaziokohler/icp/blob/master/icp.py
    # used for initial estimation of the R, t, this will make matches closer to the fact
    def svd_solver(self, pcloud_src, pcloud_dest):
        N = pcloud_src.shape[0]
        M = pcloud_src.shape[1]
        assert (pcloud_src.shape == pcloud_dest.shape)

        # substract the clouds to origins
        centroid_src = np.mean(pcloud_src, axis=0).reshape((3, 1))
        centroid_dest = np.mean(pcloud_dest, axis=0).reshape((3, 1))

        print("[ICP.svd_solver] centroid_src.shape", centroid_src.shape)

        pcl_src = pcloud_src - centroid_src.T
        pcl_dest = pcloud_dest - centroid_dest.T

        # get rotation matrix using SVD
        H = np.dot(pcl_src.T, pcl_dest)
        U, S, VT = np.linalg.svd(H)
        R = np.dot(VT.T, U.T)

        if np.linalg.det(R) < 0:
            VT[M - 1, :] *= -1
            R = np.dot(VT.T, U.T)

        # get translation
        t = centroid_dest - R.dot(centroid_src)
        print("[ICP.svd_solver] t.shape", t.shape)
        return R, t

    # @todo : TODO
    # refine the cost by obj=([R|t]*x - x').T.dot([R|t]*x - x')
    def computeError(self, pcloud_src, pcloud_dest, estimated_pose):
        N = pcloud_src.shape[0]
        assert (N == pcloud_dest.shape[0])

        # searching direction
        numOfvar = estimated_pose.shape[0]
        assert (numOfvar == 6)
        delta = np.zeros((numOfvar,))

        # reformed input functions
        def costImpl(x):
            return self.costImpl(x, pcloud_src, pcloud_dest)

        #
        err0 = costImpl(estimated_pose)
        print("[ICP] error %s for estimated_pose %s" % (
            err0,
            estimated_pose
        ))

        # using Lie algebra to computed SO3 jacobian and hessian instead, see G2OICP
        #     def jac(x):
        #       return self.jac(x, pcloud_src, pcloud_dest)
        #       pass

        #     def hess(x):
        #       return self.hess(x, pcloud_src, pcloud_dest)
        #       pass

        #     res = minimize(costImpl, estimated_pose, method='Newton-CG', jac=jac, hess=hess)

        if not self.USE_GRADIENT:
            res = fmin(costImpl, estimated_pose, full_output=1)
        else:
            if self.HAND_MADE_OPT:
                # compute se3 jaccoby internally
                res = self._minimize(pcloud_src, pcloud_dest, estimated_pose.copy(), max_iter=10)  # optimize！
                pass
            else:
                raise Exception("Not Implemented yet!")

        new_estimated_pose, err1 = res[0], res[1]
        print("[ICP] error %s with new_estimated_pose %s" % (
            err1,
            new_estimated_pose
        ))
        if np.abs(err1) < np.abs(err0):
            delta = new_estimated_pose  # - estimated_pose
            cost = err1
        else:
            print("[ICP] reject the proposal, using previous estimated_pose %s with error %f instead" % (
                estimated_pose,
                err0
            ))
            delta = estimated_pose
            cost = err0

        # reform the output
        return delta, cost

    def _get_pose(self, estimated_pose):
        self.euler.update(*estimated_pose[0:3])
        R = self.euler.R
        t = estimated_pose[3:6]
        t = t.reshape((3, 1))
        return R[:3, :3], t

    # credits to
    #  1. (2D/python) https://github.com/agnivsen/icp as the starter of the implementation
    #  2. (2D/python) https://gist.github.com/ecward/c373932638fd04a2243e
    #  3. (3D/c++) Alex Segal original G-ICP implementation :
    #  4. (2D/python) python implementation by Jacob Everist recommended by Alex Segal though it is very poor, http://jacobeverist.com/gen_icp
    def _icp_point_point(self, pcloud_src, pcloud_dest, matches, estimated_pose, callback=None):

        # make initial estiamtion
        if estimated_pose is None:
            R, t = self.svd_solver(pcloud_src, pcloud_dest)
            print("estimated_pose, \nR:\n%s\nt:\n%s\n" % (R, t))
            self.euler = Euler.fromMatrix(R)
            estimated_pose = np.array([
                self.euler.roll,
                self.euler.yaw,
                self.euler.pitch,
                t[0][0],
                t[1][0],
                t[2][0]])

            pcloud_src = self.transform(pcloud_src, R, t)

        k = 0
        numOfvars = estimated_pose.shape[0]
        while k < self.max_iterations:
            init_pose = np.zeros((numOfvars,))
            # association

            if matches is None:
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pcloud_dest)
                dist, indices = nbrs.kneighbors(pcloud_src, return_distance=True)
                print("indices: \n%s\n" % indices)
                N = pcloud_src.shape[0]
                matches = np.zeros((N, 2)).astype(np.integer)

                for i in range(N):
                    matches[i, 0] = i
                    matches[i, 1] = indices[i][0]
                pass

                print("Association: \nmatches:\n%s\n" % matches)

            # iteration
            delta, cost = self.computeError(pcloud_src[matches[:, 0].ravel()], pcloud_dest[matches[:, 1].ravel()],
                                            init_pose)

            delta = delta.reshape((6,))
            R, t = self._get_pose(delta)

            # update source point cloud
            pcloud_src = self.transform(pcloud_src, R, t)

            # update estimated pose
            estimated_pose += delta

            # check errors

            if callback is not None:
                callback(pcloud_src)

            k += 1

            print("cost:", cost)
            if np.abs(cost) < 1e-6:
                break
            else:
                print("iterating ...")

        #
        return estimated_pose

    # I am concerned that we don't have enough points and the assumption of ICP point-plane is no longer held.
    def _icp_point_plane(self):
        raise Exception("Not Implemented Yet")


# Lie Algebra ICP solver
class G2OICP: pass


class SacModel(object):

    def __init__(self):
        pass


# The algorithm was first implemented by Lei (yiak.wy@gmail.com) in C++ in later of 2019 and reimplemented in python in 2020
# you should not use this algorithm without consent of Lei in any form and purposes.
# ALL RIGHTS RESERVED

#  Points are very sparse, we don't have to do random sampling
class SacVolume(SacModel):

    def __init__(self):
        pass

class Ransac:

    def __init__(self, model):
        self._sac_model = model
        self.max_iterations = 10

    def optimize(self):
        pass