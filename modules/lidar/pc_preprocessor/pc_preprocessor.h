//
// Created by yiak on 2021/4/29.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_PC_PREPROCESSOR_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_PC_PREPROCESSOR_H

#include <memory>
#include <string>

#include <unordered_map>
using std::unordered_map;
using std::pair;

#include <thread>
#include <mutex>
/*
#include <shared_mutex> // available in c++14
 */
#include <condition_variable>
#include <functional>
#include <future>

#include <boost/filesystem.hpp>

// pcl
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
// This is the baseline algorithm, and we are going to use my new algorithm c
#include <pcl/segmentation/sac_segmentation.h>

#include <pose_graph/frame.h>
#include <base/io/sensors/velodyne_points.h>

namespace svso {
namespace lidar {

using namespace svso::base::io::reader;

struct PointCloudPreprocessorInitOptions {
    std::string sensor_name = "RawJaguar"; //need options
    std::string series_type = "RawJaguar100";
    bool is_imu_inside = true;
};

struct PointCloudPreprocessorOptions {
    // LIDAR-IMU extrinsic values for external IMU
    Eigen::Affine3d sensor2novatel_extrinsics;
    // used for motion compensation, this is especially important for highway mapping
    // typically duration for a sweep 0.1s both for Livox and traditional Velodyne VLP series
    double delta = 0.1;
    bool use_wheel_odem = true;
    bool linear_interpolation = true;
};


class PointCloudPreprocessor {
public:
    using Type = PointCloudPreprocessor;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    using PCLPoint = Point3D;
    using LidarFrame = svso::pose_graph::LidarFrame<PCLPoint>;

    PointCloudPreprocessor() { Init(); };
    bool Init();

    bool self_calibrate(typename LidarFrame::Ptr frame);

    int get_ground_plane(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                         pcl::ModelCoefficients::Ptr& coefficients,
                         double thresh);

    inline Eigen::Matrix3d Sensor2SelfCalib() {return trans_;};

    void set_use_last_guess(bool use_last_guess) {
        use_last_guess_ = use_last_guess;
    }

protected:
    void _update_coefficents(pcl::ModelCoefficients::Ptr& last_guess) {
        *coefficients_groud_plane_ = *last_guess;

        valid_guess_ = true;
        lidar_ground_plane_height_ = coefficients_groud_plane_->values[3];

    }

    enum class SupportedGroundSegmentationAlgorithm {
        UNDEF = -1,
        PCL_SAC_SEGMENTATION, // 33 ms
        FAST_LINEFIT_GROUND_SEGMENTATION, // 7 ms
        SEQUENTIAL_SAC_SEGMENT, // 10 ms
        FAST_SEQUENTIAL_LINEFIT_SAC_SEGMENTION, // 7 ms, automatically applied
    };

    SupportedGroundSegmentationAlgorithm ground_segmentation_algorithm_ =
            SupportedGroundSegmentationAlgorithm::FAST_LINEFIT_GROUND_SEGMENTATION;
private:
    Eigen::Matrix3d self_calib_rot_matrix_, device_rot_matrix_;
    Eigen::Matrix3d self_calib_rot_matrix_inverse_,  device_rot_matrix_inverse_;

    // ground parameters and ground extractor state monitors
    bool use_last_guess_ = true;
    bool valid_guess_ = false;
    int max_ground_points_ = -1;
    unsigned int frame_count_ = 0;
    unsigned int update_count_ = 0;
    bool is_ground_height_adjusted_ = false;
    float lidar_ground_plane_height_ = 0.0;

    pcl::ModelCoefficients::Ptr coefficients_groud_plane_;

    typename LidarFrame::SensorPointsPtr sensor_points_;

    // p_calibrated = self_calib_rot_matrix_ * device_rot_matrix_ * p_sensor
    Eigen::Matrix3d trans_;
};

} // lidar
} // mapping

#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_PC_PREPROCESSOR_H
