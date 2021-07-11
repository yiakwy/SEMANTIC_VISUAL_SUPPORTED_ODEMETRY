//
// Created by yiak on 2021/7/3.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_SEQUENTIAL_LINEFIT_FLAGS_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_SEQUENTIAL_LINEFIT_FLAGS_H

#include <gflags/gflags.h>

namespace svso {
namespace lidar {
namespace perception {

// local debugger control
DECLARE_bool(debug_lidar_perception);

// ransac controls
DECLARE_double(dist_thresh);
DECLARE_int32(max_iterations);
DECLARE_int32(sequential_max_iterations);

    } // perception
  } // lidar
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_SEQUENTIAL_LINEFIT_FLAGS_H
