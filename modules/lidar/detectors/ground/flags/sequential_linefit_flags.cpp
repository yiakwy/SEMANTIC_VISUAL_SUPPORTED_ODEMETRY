//
// Created by yiak on 2021/7/5.
//

#include "sequential_linefit_flags.h"

namespace mapping {
namespace lidar {
namespace perception {

DEFINE_bool(debug_lidar_perception, true, "lidar perception module debugger switch");
DEFINE_double(dist_thresh, 0.1, "default RanSAC distance");
DEFINE_int32(max_iterations, 50, "default maximum iterations");
DEFINE_int32(sequntial_max_iterations, 20, "default maximum iterations");

    } // perception
  } // lidar
} // svso