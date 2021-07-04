//
// Created by yiak on 2021/7/1.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_POSE_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_POSE_H

#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "base/logging.h"
#include "base/exceptions.h"

namespace svso {
namespace base {
namespace math {

using namespace base::logging;
using namespace base::exceptions;

// On ground vehicle can be reduced to a 3 dimension problem
class Pose3DoF;
class Pose6DoF {
public:
    using Type = Pose6DoF;
    using Ptr = std::shared_ptr<Type>;

    Pose6DoF() {}
    virtual ~Pose6DoF() {}

    void Init() {

    }

    /*
     * attributes
     */
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    Eigen::Quaterniond q;

    // internal representation of the pose
    // transformation from local to world
    Eigen::Matrix4d Tr_world_local;
    Eigen::Isometry3d _pose;

    // will be used in 3 DoF pose graph optimization
    std::shared_ptr<Pose3DoF> light_pose;

    void set_pos(const Eigen::Vector3d& pos) {
        t = pos;

        // update data buffer to update :
        x = t[0];
        y = t[1];
        z = t[2];

        // Tr_world_local
        Tr_world_local.col(3) << x, y, z, 1;

        // _pose
        _pose = Eigen::Isometry3d::Identity();
        _pose.rotate(R);
        _pose.pretranslate(t); // equivalent to v'=R*v + t

        update(true, false);
    }

    void set_rot(const Eigen::Quaterniond& rot) {
        q = rot;

        // update SE3 data buffer
        //

        // Euler ZYX angles, returned angles are in the ranges [0:pi]x[-pi:pi]x[-pi:pi]
        auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
        roll = euler[0];
        pitch = euler[1];
        yaw = euler[2];

        // use SE3 data buffer or quaternion to update :
        //

        // Euler rotation axis and theta values
        Eigen::Vector3d axis = q.vec();
        double sin_theta = axis.norm();
        axis /= sin_theta;
        double cos_theta = q.w();
        double theta = acos(cos_theta); // restrict to [0, pi]
        if (theta > M_PI / 2) {
            LOG(INFO) << format("[Pose6DoF::set_rot] the rotation theta (%f) should not be great > M_PI / 2 (%f)", theta, M_PI/2);
        }
        if (sin_theta < 0) {
            theta = M_PI / 2 - theta;
        }

        // R
        R = q.toRotationMatrix();

        // Tr_world_local
        Tr_world_local.block<3,3>(0,0) = R;

        // _pose
        _pose = Eigen::Isometry3d::Identity();
        _pose.rotate(R);
        _pose.pretranslate(t); // equivalent to v'=R*v + t

        update(false, true);
    }

    // update all variables :
    // @todo TODO
    void update(bool update_euclidean_trans=true, bool update_so3_trans=true) {
        if (update_euclidean_trans) {
            do_update_euclidean_trans();
        }
        if (update_so3_trans) {
            do_update_so3_trans();
        }
    }

    void do_update_euclidean_trans() {

    }

    void do_update_so3_trans() {

    }

    const double* data() const {
        return data_;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    // SE3 data buffer
    union {
        double data_[6];
        struct {
            double x;
            double y;
            double z;
            // three Euler angles
            double roll;  // X axis
            double pitch; // Y axis
            double yaw;   // Z axis
        };
    };
};

class Pose3DoF {
public:
    Pose3DoF() {}
    virtual ~Pose3DoF() {}

    explicit Pose3DoF(Eigen::Vector3d& axis, double rot_theta) {
        axis_ = axis;
        theta = rot_theta;
    }

    /*
     * attributes
     */

public:
    // SE2 data buffer
    Eigen::Vector3d axis_;
    double height_;
    union {
        double data_[3];
        struct {
            double x;
            double y;
            double theta; // typically yaw angle
        };
    };

};

    } // math
  } // base
} // svso

#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_POSE_H
