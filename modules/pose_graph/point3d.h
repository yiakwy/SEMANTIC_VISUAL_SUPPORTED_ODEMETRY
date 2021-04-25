//
// Created by yiak on 2020/7/29.
//
#pragma once

#ifndef SEMANTIC_RELOCALIZATION_POINT3D_H
#define SEMANTIC_RELOCALIZATION_POINT3D_H

#include <iostream>
#include <memory>
using std::static_pointer_cast;

#include <vector>
using std::vector;

#include <set>
using std::set;

#include <string>

// used for RangeImage representation
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
// some useful ISP libraries
#include <opencv2/imgproc.hpp>

#ifndef NDEBUG
#include <opencv2/highgui.hpp>
#endif

#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif
// #include <pcl/pcl_memory.h>
// #include <pcl/pcl_macros.h>

#include <pcl/pcl_base.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
// #include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/ModelCoefficients.h>

#include <math.h>
#include <cmath>

#include <ctime>
#include <climits>

#include <thread>

// parallel container
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_hash_map.h>

// logging
#include <base/logging.h>
#include <base/exceptions.h>

namespace svso {
namespace pose_graph {

using namespace svso::base::logging;
using namespace svso::base::exceptions;



} // pose_graph
} // svso

#endif //SEMANTIC_RELOCALIZATION_POINT3D_H
