//
// Created by LEI WANG on 2021/7/1.
//
#pragma once

#include <stdarg.h>  // C arbitrary arguments
#include <string.h>  // C string library
#include <stdint.h>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <locale>
#include <utility> // std::is_same, std::enable_if

#include <vector>
#include <iterator> // needed for std::ostream_iterator

#include <Eigen/Dense>
#include <cmath>

#include <sstream>
#include <string>

#include "glog/logging.h"

namespace Eigen {
template <typename Scalar>
using MatrixX = Matrix<Scalar, Dynamic, Dynamic>;

template<typename Scalar>
using VectorX = Matrix<Scalar, Dynamic, 1>;
}

namespace svso {
namespace base {
namespace logging {

using std::string;
using std::ostringstream;

#ifndef __BUF_SIZE
#define __BUF_SIZE 1024
#endif

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

// FileLogger


// *** logging utitlity for vector
template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    if (!v.empty()) {
        out << '[';
        std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
}

// ***
std::string extract_module_name(const std::string& short_file_name);

// eigen logging utils
class EigenMatrixFormatter {
public:
    template<typename Type>
    static std::string pretty_print(const ::Eigen::MatrixX<Type>& mat) {
        std::stringstream stream;
        ::Eigen::IOFormat HeavyFmt(::Eigen::FullPrecision, 0, " ", ";\n  ", "[ ", " ]", "[ ", " ]\n");
        stream << mat.format(HeavyFmt);
        return stream.str();

    }

    template<typename Type>
    static std::string pretty_print(const ::Eigen::MatrixX<Type>&& mat) {
        return pretty_print(mat);
    }

    template<typename Type>
    static std::string pretty_print(const ::Eigen::Block<::Eigen::MatrixX<Type>>&& mat) {
        typename ::Eigen::MatrixX<Type> tmp_mat = mat;
        return pretty_print(tmp_mat);
    }

    template<typename Type>
    static std::string pretty_print(const ::Eigen::VectorX<Type>& vec)
    {
        typename ::Eigen::MatrixX<Type> mat(vec.rows(), 1);
        mat.col(0) = vec;
        return pretty_print(mat);
    }

};

// *** special formmater
#define INT64_C_FMT PRIu64

    } // logging
  } // base
} // svso
