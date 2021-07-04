//
// Created by yiak on 2021/7/1.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_ARRAY_LIKE_HASHER_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_ARRAY_LIKE_HASHER_H

#include <unordered_map>
using std::pair;

#include <functional>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "base/exceptions.h"

namespace svso {
namespace base {

using namespace svso::base::exceptions;

static const long long SEED=0x9e3779b9;

// array-alike hasher, will be used in our LidarPoints Indexer : RangeImage(VelodynePoints), VoxelGrid and OccupancyGrid
struct _hasher {

    template<typename T>
    std::size_t operator()(T const & shape,
            typename std::enable_if<std::is_same<T,
                                                 std::vector<typename T::value_type,
                                                             typename T::allocator_type>>::value, T>::type* = nullptr) const
    {
        using IndexScalar = typename T::value_type;
        size_t ret = 0;
        for (size_t i=0; i < shape.size(); i++)
        {
            ret ^= std::hash<IndexScalar>()( shape[i] ) + SEED + (ret << 6) + (ret >> 2);
        }
        return ret;
    }

    template<typename T>
    std::size_t operator()(T const & shape,
            typename std::enable_if<std::is_same<T, Eigen::Matrix<typename T::Scalar, -1, 1>>::value, T>::type* = nullptr)
    {
        using IndexScalar = typename T::Scalar;
        size_t ret = 0;
        for (size_t i=0; i < shape.size(); i++)
        {
            ret ^= std::hash<typename T::Scalar>()( shape(i) ) + SEED + (ret << 6) + (ret >> 2);
        }
        return ret;
    }

    template<typename T>
    std::size_t operator()(T const & shape,
            typename std::enable_if<std::is_same<T,
                                                 std::pair<typename T::first_type,
                                                           typename T::second_type>>::value, T>::type* = nullptr)
    {
        size_t ret = 0;
        ret ^= std::hash<typename T::first_type>()( (&shape)->first  ) + SEED + (ret << 6) + (ret >> 2);
        ret ^= std::hash<typename T::second_type>()( (&shape)->second ) + SEED + (ret << 6) + (ret >> 2);
        return ret;
    }

    /*
    std::size_t operator()(T const & shape) const {
        using IndexScalar = int;
        size_t ret = 0;

        if (std::is_same<T, Eigen::Tensor<IndexScalar, 1>>::value) {
            typename T::Dimensions dims = shape.dimensions();
            for (size_t i=0; i < dims[0]; i++) {
                ret ^= std::hash<typename T::Scalar>()( shape(i) ) + SEED + (ret << 6) + (ret >> 2);
            }
        } else
        if (std::is_same<T, Eigen::Matrix<IndexScalar, Eigen::Dynamic, 1>>::value) {
            for (size_t i=0; i < shape.size(); i++)
            {
                ret ^= std::hash<typename T::Scalar>()( shape(i) ) + SEED + (ret << 6) + (ret >> 2);
            }
        } else

        if (std::is_same<T, std::pair<IndexScalar, IndexScalar>>::value) {
            ret ^= std::hash<IndexScalar>()( (&shape)->first  ) + SEED + (ret << 6) + (ret >> 2);
            ret ^= std::hash<IndexScalar>()( (&shape)->second ) + SEED + (ret << 6) + (ret >> 2);
        }
        // treat T as general array
        else {
            for (size_t i=0; i < shape.size(); i++)
            {
                ret ^= std::hash<typename T::Scalar>()( shape(i) ) + SEED + (ret << 6) + (ret >> 2);
            }
        }

        return ret;
    }
    */
};

template<typename T>
struct _pair_hasher {
    // usd for intel tbb internally
    static std::size_t hash(T const & shape) {
        using IndexScalar = int;
        size_t ret = 0;
        if (std::is_same<T, std::pair<IndexScalar, IndexScalar>>::value) {
            // Cantor pairing function:
            ret = (shape.first + shape.second) * (shape.first + shape.second + 1) / 2 + shape.first;
        } else {
            NOT_IMPLEMENTED
        }

        return ret;
    }

    static bool equal (T const & left_shape, T const & right_shape) {
        using IndexScalar = int;
        if (std::is_same<T, std::pair<IndexScalar, IndexScalar>>::value) {
            if (left_shape.first == right_shape.first && left_shape.second == right_shape.second) {
                return true;
            } else {
                return false;
            }
        }
        else {
            NOT_IMPLEMENTED
        }
    }
};

  } // base
} // svso

#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_ARRAY_LIKE_HASHER_H
