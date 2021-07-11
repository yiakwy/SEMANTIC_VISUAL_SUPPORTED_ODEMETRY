//
// Created by yiak on 2021/7/5.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_LINEFIT_GROUND_SEGMENT_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_LINEFIT_GROUND_SEGMENT_H

#include <memory>
#include <mutex>

#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "base/io/sensors/velodyne_points.h"
#include "lidar/detectors/ground/linefit_ground_segment/segment.h"

namespace svso {
namespace lidar {
namespace perception {

using namespace svso::base::io::reader;

// use our registered PCL Vec3 and custom Point3D
using PCLPoint = Point3D;
using PointCloud = pcl::PointCloud<PCLPoint>;
using PointLine = std::pair<PCLPoint, PCLPoint>;

struct LinefitGroundSegmentInitOptions {

    /*
     * Ground structure
     */

    // Number of radial bins.
    int n_bins = 120;

    // Number of angular segments.
    int n_segments = 360;

    // Minimum range of segmentation.
    double r_min_square = 0.5 * 0.5;
    double r_min = 0.5;

    // Maximum range of segmentation.
    double r_max_square = 80 * 80;
    double r_max = 80;

    LivoxPointsInitOptions livox_points_init_options;

    /*
     * ** Ground conditions **
     */

    // Height of sensor above ground.
    double sensor_height = 0.3;

    // Maximum distance to a ground line to be classified as ground.
    double max_dist_to_line = 0.05;

    // Max slope to be considered ground line.
    double max_slope = 0.3;

    // Max error for line fit.
    double max_error_square = 0.05 * 0.05;

    // Maximum height of starting line to be labelled as ground.
    double max_start_height = 0.2;

    // Distance at which points are considered far from each other.
    double long_threshold = 1.0;

    // Maximum slope for
    double max_long_height = 0.1;

    // How far to search for a line in angular direction [rad].
    double line_search_angle = 0.1;

    /*
     * ** Device parameters **
     */
    int n_threads_used_for_linefit = 4;
};

class LinefitGroundSegment {
public:
    using Type = LinefitGroundSegment;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    LinefitGroundSegment(const LinefitGroundSegmentInitOptions &init_options = LinefitGroundSegmentInitOptions());
    virtual ~LinefitGroundSegment() { Clear(); }

    void Clear() {}

    void segment(const PointCloud &cloud, std::vector<int> *segmentation);

private:
    void insertPoints(const PointCloud &cloud);

    void fitLines();

    void assignCluster(std::vector<int> *segmentation);

private:
    const LinefitGroundSegmentInitOptions init_options_;

    // Access with segments_[segment][bin].
    std::vector <Segment> segments_;

    // Bin index of every point.
    std::vector <std::pair<int, int>> bin_index_;

    // 2D coordinates (d, z) of every point in its respective segment.
    std::vector <Bin::MinZPoint> segment_coordinates_;

};


    } // perception
  } // lidar
} // mapping
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_LINEFIT_GROUND_SEGMENT_H
