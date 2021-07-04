//
// Created by LEI WANG on 2021/7/1.
//
#include "logging.h"
#include "env_config.h"

using namespace svso::system;

namespace svso {
namespace base {
namespace logging {

void Init_GLog(int argc, const char** argv) {
  // fLS is not defined in glog
  FLAGS_logtostderr = true;

  // Set log directory
  google::SetLogDestination(
          google::INFO, C_MAPPING_FORMAT("%s/info.log", env_config::LOG_DIR.c_str()).c_str());
  google::SetLogDestination(
          google::ERROR, C_MAPPING_FORMAT("%s/error.log", env_config::LOG_DIR.c_str()).c_str());

  // init glog
  google::InitGoogleLogging(argv[0]);

  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;
}

string format(const char* fmt, ...) {
  va_list ap;
  char buf[__BUF_SIZE];
  string ret;

  va_start(ap, fmt);
  vsprintf(buf, fmt, ap);
  va_end(ap);

  ret = buf;
  return ret;
}

std::string extract_module_name(const std::string& short_file_name) {
    std::string ret = short_file_name;
    const char* ptr = ret.c_str();
    int end = strchr(ptr, '.') - ptr;
    return ret.substr(0, end);
}

    } // logging
  } // base
} // svso
