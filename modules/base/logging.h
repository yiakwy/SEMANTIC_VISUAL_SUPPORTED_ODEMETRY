//
// Created by LEI WANG on 19-8-28.
//
#pragma once

#include <stdarg.h>  // C arbitrary arguments
#include <string.h>  // C string library
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <locale>
#include "glog/logging.h"

using std::string;
using std::ostringstream;

#define BUF_SIZE 1024

namespace svso {
namespace base {
namespace logging {

#define Mapping_format(fmt, ...) format(fmt, ##__VA_ARGS__)

void Init_GLog(int argc, const char** argv);

string format(const char* fmt, ...);

string format(string fmt, ...);

    }
  }
}
