//
// Created by yiak on 2021/7/5.
//  Adapted from original codebase https://github.com/lorenwel/linefit_ground_segmentation
//  to replace PCL SACSegmentation algorithm
// Credits to original author
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_BIN_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_BIN_H

#include <atomic>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <math.h>
#include <cmath>

#include "base/io/sensors/velodyne_points.h"

namespace svso {
namespace lidar {
namespace perception {

using namespace svso::base::io::reader;

// use our registered PCL Vec3 and custom Point3D
using PCLPoint = Point3D;

#define EPILON 1e-3

// we prefer use Segment instead of Bin in program while keep original name style from Linefit Ground Segmentation method
// i.e., <segment, bin> -> <sector, segment>
class Bin {
public:
    struct MinZPoint {
        MinZPoint() : d(0), z(0) {}
        MinZPoint(const double& d, const double& z) : d(d), z(z) {}
        bool operator==(const MinZPoint& other) {
            if (abs(d - other.d) < EPILON && abs(z - other.z) < EPILON) {
                return true;
            } else {
                return false;
            };
        }

        // always tracking the minimum (d, z) coordinates in this bin
        /*
         * data attributes
         */
        double d;
        double z;
    };

public:

    Bin();

    /// \brief Fake copy constructor to allow vector<vector<Bin> > initialization.
    Bin(const Bin& bin);

    void addPoint(const PCLPoint& point);

    void addPoint(const double& d, const double& z);

    MinZPoint getMinZPoint();

    inline bool hasPoint() {return has_point_;}

private:
    std::atomic<bool> has_point_;
    std::atomic<double> min_z;
    std::atomic<double> min_z_range;
};


    } // perception
  } // lidar
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_BIN_H
