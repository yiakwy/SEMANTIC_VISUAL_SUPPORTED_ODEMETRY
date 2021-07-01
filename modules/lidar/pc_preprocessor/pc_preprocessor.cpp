//
// Created by yiak on 2021/4/29.
//

#include "pc_preprocessor.h"

namespace svso {
namespace lidar {

bool
PointCloudPreprocessor::Init() {
    NOT_IMPLEMENTED
}

bool
PointCloudPreprocessor::self_calibrate(typename LidarFrame::Ptr frame) {
    NOT_IMPLEMENTED
}

int
PointCloudPreprocessor::get_ground_plane(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, pcl::ModelCoefficients::Ptr& coefficients, double thresh) {
    NOT_IMPLEMENTED
}

} // lidar
} // svso