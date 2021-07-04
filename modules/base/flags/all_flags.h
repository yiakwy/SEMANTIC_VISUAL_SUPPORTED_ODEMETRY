//
// Created by yiak on 2021/7/1.
//

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_ALL_FLAGS_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_ALL_FLAGS_H

#include <gflags/gflags.h>

namespace svso {
namespace base {

// discard usage of CMAKE micro "NDEBUG" in favor of gflags control
DECLARE_bool(all_debug);

  } // base
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_ALL_FLAGS_H
