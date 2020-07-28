//
// Created by yiak on 2020/7/23.
//

#ifndef SEMANTIC_RELOCALIZATION_EIGEN_UTIL_H
#define SEMANTIC_RELOCALIZATION_EIGEN_UTIL_H

#include <Eigen/Dense>
#include <cmath>

#include <sstream>
#include <string>


namespace svso {
    namespace models {
        namespace eigen_utils {

class Eigen {
public:
    static std::string pretty_print(::Eigen::VectorXd& vec)
    {
        ::Eigen::MatrixXd mat(vec.rows(), 1);
        mat.col(0) = vec;
        std::stringstream stream;
        ::Eigen::IOFormat HeavyFmt(::Eigen::FullPrecision, 0, " ", ";\n  ", "[ ", " ]", "[ ", " ]\n");
        stream << mat.format(HeavyFmt);
        return stream.str();
    }

    static std::string pretty_print(::Eigen::MatrixXd& mat)
    {
        std::stringstream stream;
        ::Eigen::IOFormat HeavyFmt(::Eigen::FullPrecision, 0, " ", ";\n  ", "[ ", " ]", "[ ", " ]\n");
        stream << mat.format(HeavyFmt);
        return stream.str();
    }
};

        }
    }
}

#endif //SEMANTIC_RELOCALIZATION_EIGEN_UTIL_H
