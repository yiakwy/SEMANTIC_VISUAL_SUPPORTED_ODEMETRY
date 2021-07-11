//
// Created by yiak on 2021/7/5.
//

#include "bin.h"

namespace svso {
namespace lidar {
namespace perception {

Bin::Bin() : min_z(std::numeric_limits<double>::max()), has_point_(false) {}

Bin::Bin(const Bin& bin) : min_z(std::numeric_limits<double>::max()),
                           has_point_(false) {}

void Bin::addPoint(const PCLPoint& point) {
    const double d = sqrt(point.x * point.x + point.y * point.y);
    addPoint(d, point.z);
}

void Bin::addPoint(const double& d, const double& z) {
    has_point_ = true;
    if (z < min_z) {
        min_z = z;
        min_z_range = d;
    }
}

Bin::MinZPoint Bin::getMinZPoint() {
    MinZPoint point;

    if (has_point_) {
        point.z = min_z;
        point.d = min_z_range;
    }

    return point;
}

    } // perception
  } // lidar
} // svso