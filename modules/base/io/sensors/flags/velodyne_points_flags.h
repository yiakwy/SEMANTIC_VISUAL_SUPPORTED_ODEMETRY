//
// Created by yiak on 2021/7/1.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_VELODYNE_POINTS_FLAGS_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_VELODYNE_POINTS_FLAGS_H

#include <gflags/gflags.h>

namespace svso {
namespace base {
namespace io {
namespace reader {

DECLARE_bool(use_ostu_light_equalization);

      } // reader
    } // io
  } // base
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_VELODYNE_POINTS_FLAGS_H
