//
// Created by yiak on 2021/4/30.
//

#include "config_manager_flags.h"

namespace svso {
namespace base {
namespace io {

DEFINE_string(conf_root, "config_root.yml", "config root file");
DEFINE_string(plus_ai_calib_db_root, "/opt/plusai/var/calib_db/", "calibration file root");

    } // io
  } // base
} // svso