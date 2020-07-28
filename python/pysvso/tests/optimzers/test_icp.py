import os, sys
import unittest
from pysvso.optimizers.icp import ICP
from pysvso.lib.maths.rotation import Euler

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


def plot_data(X1, X2, Z, ax=None, color='b'):
    if ax is None:
        fig = plt.figure(dpi=60)
        ax = fig.gca(projection='3d')
    ax.scatter(X1, X2, Z, c=color, marker='o', s=64)

    for i, x, y, z in zip(range(len(X1)), X1, X2, Z):
        p, q, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        # ax.annotate("%s" % i, xy=(p,q), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))#arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
        # ax.annotate("%s" % i, xy=(p,q))
        # ax.text(x,y,Z[i], '%s' % (str(i)), size=20, color='red')
    # plt.show()
    return ax

def setup_data():
    # test data
    ang = np.linspace(-np.pi / 2, np.pi / 2, 10)
    z = np.zeros((ang.shape[0],))
    pcloud_src = np.array([ang, np.sin(ang), z]).T
    print("pcloud_src shape: ", pcloud_src.shape)
    print(pcloud_src[:, 2].shape)

    REAL_POSE = np.array([0, 0, np.pi / 2, 0.2, 0.3, 0])
    R, t = _get_pose(REAL_POSE)
    pcloud_dest = (R.dot(pcloud_src.T) + t).T
    return pcloud_src, pcloud_dest

pcloud_src, pcloud_dest = setup_data()

def user_plot(pcloud):
    X1 = pcloud[:, 0]
    X2 = pcloud[:, 1]
    Z = pcloud[:, 2]

    ax = plot_data(X1, X2, Z)
    ax = plot_data(pcloud_dest[:, 0], pcloud_dest[:, 1], pcloud_dest[:, 2], ax=ax, color='r')

    plt.show()

def _get_pose(estimated_pose):
    euler = Euler(*estimated_pose[0:3])
    euler.update(*estimated_pose[0:3])
    R = euler.R
    t = estimated_pose[3:6]
    t = t.reshape((3, 1))
    return R[:3, :3], t

def test_icp():
    ax = plot_data(pcloud_src[:, 0], pcloud_src[:, 1], pcloud_src[:, 2])
    ax = plot_data(pcloud_dest[:, 0], pcloud_dest[:, 1], pcloud_dest[:, 2], ax=ax, color='r')

    plt.show()

    # test and visualize

    # RUN icp ALGORITHM
    icp = ICP()

    icp._icp_point_point(pcloud_src, pcloud_dest, None, None, callback=user_plot)