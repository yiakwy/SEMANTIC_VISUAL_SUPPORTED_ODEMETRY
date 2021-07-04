//
// Created by yiak on 2021/7/1.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_CAMERA_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_CAMERA_H

#include <memory>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <base/logging.h>
#include "base/exceptions.h"

#include "base/math/angles.h"
#include "base/math/pose.h"


namespace svso {
namespace pose_graph {

using namespace svso::base::logging;
using namespace svso::base::exceptions;
using namespace svso::base::math;

// used to load description of devices
class Device {
public:
    using Type = Device;
    using Ptr = std::shared_ptr<Device>;
    using ConstPtr = std::shared_ptr<const Device>;

    Device() {}

    void load(const std::string& device_path) {
        cv::FileStorage device_config(device_path, cv::FileStorage::READ);
        CHECK(device_config.isOpened()) << format("Failed to open yaml file : %s", device_path.c_str());

        cv::FileNode root = device_config.root();

        if (!root["LiDAR"].isNone() && !root["LiDAR"].empty()) {
            cv::FileNode lidar_node = root["LiDAR"];
            cv::Mat T_imu_lidar_tmp;
            lidar_node[Tr_lidar_to_imu_key] >> T_imu_lidar_tmp;
            T_imu_lidar = Eigen::Matrix4d::Identity(4,4);
            cv::cv2eigen(T_imu_lidar_tmp, T_imu_lidar);
        }

        if (!root["Camera"].isNone() && !root["Camera"].empty()) {
            cv::FileNode camera_node = root["Camera"];
            cv::Mat distortion_tmp;
            fx = camera_node["fx"];
            fy = camera_node["fy"];
            cx = camera_node["cx"];
            cy = camera_node["cy"];
            camera_node["distortion"] >> distortion_tmp;
            distortion = Eigen::Matrix<double, 4, 1>::Identity(4, 1);
            cv::cv2eigen(distortion_tmp, distortion);

            K = Eigen::Matrix3d::Identity();
            K[0,0] = fx;
            K[1,1] = fy;
            K[0,2] = cx;
            K[1,2] = cy;
        }
    }

    static std::string Tr_lidar_to_imu_key;

    // Lidar to IMU matrix
    Eigen::Matrix4d T_imu_lidar;

    // intrinsic parameters for camera
    double fx;
    double fy;
    double cx;
    double cy;
    Eigen::Vector<double, 5> distortion;

    Eigen::Matrix3d K;
};

class Camera {
public:
    using Type = Camera;
    using Ptr = std::shared_ptr<Camera>;
    using ConstPtr = std::shared_ptr<const Camera>;

    Camera() {}
    virtual ~Camera() {}

    enum class Mode {
        MONOCULAR = 1,
        STEREO = 2,
        DEPTHS = 3 // default LiDAR device
    };



};

  } // pose_graph
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_CAMERA_H