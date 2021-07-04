//
// Created by yiak on 2021/7/1.
//

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_ANGLES_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_ANGLES_H

#include <cmath>

namespace svso {
namespace base {
namespace math {

inline double rad2deg(double rad) {
    return rad * 180.0 / M_PI;
}

inline double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

    } // math
  } // base
} // svso

#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_ANGLES_H
