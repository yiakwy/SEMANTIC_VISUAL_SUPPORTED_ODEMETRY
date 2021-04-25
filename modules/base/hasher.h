//
// Created by yiak on 2021/4/26.
//
#pragma once

#ifndef SEMANTIC_RELOCALIZATION_HASHER_H
#define SEMANTIC_RELOCALIZATION_HASHER_H

#include <unordered_map>
using std::pair;

#include <functional>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace svso {
namespace base {

static const long long SEED=0x9e3779b9;

// array-alike hasher, will be used in our LidarPoints Indexer : RangeImage(VelodynePoints), VoxelGrid and OccupancyGrid
template<typename T>
struct _hasher : std::unary_function<T, size_t> {
    std::size_t operator()(T const & shape) const {
        using IndexScalar = int;
        size_t ret = 0;
        ///*
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
        //*/
        if (std::is_same<T, std::pair<IndexScalar, IndexScalar>>::value) {
            ret ^= std::hash<IndexScalar>()( shape.first ) + SEED + (ret << 6) + (ret >> 2);
            ret ^= std::hash<IndexScalar>()( shape.second ) + SEED + (ret << 6) + (ret >> 2);
        }
        // treat T as general array
        else {
            for (size_t i=0; i < shape.size(); i++)
            {
                ret ^= std::hash<typename T::Scalar>()( shape(i) ) + 0x9e3779b9 + (ret << 6) + (ret >> 2);
            }
        }

        return ret;
    }

};

template<typename T>
struct _pair_hasher {
    // usd for intel tbb internally
    static std::size_t hash(T const & shape) {
        using IndexScalar = int;
        size_t ret = 0;
        if (true) {// (std::is_same<T, std::pair<int,int>>::value) {
            // Cantor pairing function:
            ret = (shape.first + shape.second) * (shape.first + shape.second + 1) / 2 + shape.first;
        }

        return ret;
    }

    static bool equal (T const & left_shape, T const & right_shape) {
        using IndexScalar = int;
        if (true) {//(std::is_same<T, std::pair<IndexScalar, IndexScalar>>::value) {
            if (left_shape.first == right_shape.first && left_shape.second == right_shape.second) {
                return true;
            } else {
                return false;
            }
        }
        else {
            return false;
        }
    }
};

} // base
} // svso

#endif //SEMANTIC_RELOCALIZATION_HASHER_H
