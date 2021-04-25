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
#include <utility> // std::is_same, std::enable_if
#include "glog/logging.h"

using std::string;
using std::ostringstream;

#define BUF_SIZE 1024

namespace svso {
namespace base {
namespace logging {

#define __SHORT_FILE__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define __SHORT_FILE(NAME) (strrchr(NAME, '/')+1)

#define C_MAPPING_FORMAT(fmt, ...) format(fmt, ##__VA_ARGS__)

void Init_GLog(int argc, const char** argv);

#ifndef __LOGGING_C_FORMAT__
#define __LOOGING_C_FORMAT__
string format(const char* fmt, ...);
#endif

// avoid non-explicit conversion from char* to std::string, and of course you cannot do the vice versa @yiawy
template<typename T, typename... Args>
string format(T fmt, typename std::enable_if<std::is_same<T, std::string>::value, T>::type* = 0, Args... args)
{
    std::string ret = format(fmt.c_str(), args...);
    return ret;
}

    }
  }
}
