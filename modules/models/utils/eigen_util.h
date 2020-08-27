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

        }
    }
}

#endif //SEMANTIC_RELOCALIZATION_EIGEN_UTIL_H
