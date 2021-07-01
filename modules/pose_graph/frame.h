//
// Created by yiak on 2020/7/29.
//
#pragma once

#ifndef SEMANTIC_RELOCALIZATION_FRAME_H
#define SEMANTIC_RELOCALIZATION_FRAME_H

#include <memory>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_traits.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <base/exceptions.h>

#include <base/misc.h>
#include <base/io/velodyne_points.h>

namespace svso {
namespace pose_graph {

using svso::base::exceptions;
using svso::base::io::reader;

class Frame;
template<typename PCLPoint>
class LidarFrame;

class Frame : std::enable_shared_from_this<Frame> {
public:
    using Type = Frame;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;
    using WeakPtr = std::weak_ptr<Type>;

    using FrameKey = uint64_t;

    Frame() : identity() {}
    virtual ~Frame() {}

    // content
    bool isKeyFrame = false;
    bool isFirst = false;
    double timestamp;

    svso::base::Identity identity;
    FrameKey key;

    // Rigid object movements

    // rotation and translation relative to the origin
    Eigen::Matrix4d Rwc;

    // rotation and translation relative to the last frame
    Eigen::Matrix4d dRwc;

    // Ground Truth measured by high definition IMU
    Eigen::Matrix4d R_gt;

    // Covisibility Graph Toplogy
    std::weak_ptr<Frame> pre;

    // Features Expression Layer
    // @todo TODO KeyPoints group

    // @todo TODO
    virtual void Clear() {
        NOT_IMPLEMENTED
    }

    // @todo TODO
    virtual void Reset() {
        NOT_IMPLEMENTED
    }
};


template<typename PCLPoint>
class LidarFrame : public Frame {
public:
    using Base = Frame;
    using Type = LidarFrame;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;
    using WeakPtr = std::weak_ptr<Type>;

    using PointCluster = typename pcl::PointCloud<PCLPoint>::Ptr;
    using SensorPoints = LivoxPoints;
    using SensorPointsPtr = LivoxPoints::Ptr;

};

} // pose_graph
} // svso
#endif //SEMANTIC_RELOCALIZATION_FRAME_H
