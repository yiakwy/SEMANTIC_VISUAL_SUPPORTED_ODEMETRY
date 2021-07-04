//
// Created by yiak on 2021/7/1.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_FRAME_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_FRAME_H

#include <memory>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_traits.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "base/exceptions.h"

#include "base/misc.h"
#include "base/io/sensors/velodyne_points.h"

#include <boost/filesystem.hpp>

#include "camera.h"
#include "point3d.h"

namespace svso {
namespace pose_graph {

using namespace svso::base::exceptions;
using namespace svso::base::io::reader;
namespace fs = boost::filesystem;

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

    base::Identity identity;
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

    static base::AtomicCounter seq;

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

class ImgFrame : public Frame {
public:
    using Base = Frame;
    using Type = ImgFrame;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;
    using WeakPtr = std::weak_ptr<Type>;



private:
    std::string img_src_;
};

// PCLPoint should be coherent with PointCloudPreprocessor
template<typename PCLPoint>
class LidarFrame : public Frame {
public:
    using Base = Frame;
    using Type = LidarFrame;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;
    using WeakPtr = std::weak_ptr<Type>;

    using PointCluster = typename pcl::PointCloud<PCLPoint>::Ptr;
    using SensorPoints = VelodynePoints;
    using SensorPointsPtr = VelodynePoints::Ptr;

private:
    std::string cloud_src_;

};

  } // pose_graph
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_FRAME_H
