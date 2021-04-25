//
// Created by yiak on 2021/4/23.
//
#pragma once

#ifndef SEMANTIC_RELOCALIZATION_VEC3_H
#define SEMANTIC_RELOCALIZATION_VEC3_H

#include <iostream>
#include <memory>
using std::static_pointer_cast;

#include <Eigen/Core>
#include <Eigen/Dense>

#include <base/logging.h>
#include <base/exceptions.h>

namespace svso {
namespace base {
namespace math {

using namespace svso::base::logging;
using namespace svso::base::exceptions;

class Vec3 {
public:
    using Type = Vec3;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    Vec3() : x(0), y(0), z(0) {}
    explicit Vec3(double x, double y, double z=0) : x(x), y(y), z(z) {}
    virtual ~Vec3() { Clear(); }

    Vec3(const Vec3 &other) {
        CopyFrom(other);
    }

    Vec3(Vec3 &&other) noexcept
            : Vec3() {
        *this = ::std::move(other);
    }

    inline Vec3 &operator=(const Vec3 &other) {
        CopyFrom(other);
        return *this;
    }

    inline Vec3 &operator=(Vec3 &&other) noexcept {
        if (this != &other) {
            CopyFrom(other);
        }
        return *this;
    }

    void CopyFrom(const Vec3& other) {
        if (this == &other) return;
        Clear();
        x = other.x;
        y = other.y;
        z = other.z;
    }

    void Clear() {}

    // data field
    union {
        // PCL will use an extra field for padding with SSE-enabled processors
        float data[4];
        struct  {
            float x;
            float y;
            float z;
        };
        Eigen::Vector3f vec3f;
        Eigen::Vector2f vec2f;
    };
public:
    // align memory with eigen allocator, see Eigen document 3.3.7 : Structures Having Eigen Members
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:

    // overriden arithmetic operators
    // @see https://drake.mit.edu/styleguide/cppguide.html
    // also @see https://drake.mit.edu/styleguide/cppguide.html#Operator_Overloading
    // we will implement these later ...

    // Binary operators:
    friend Vec3 operator+(const Vec3 &lhs, const Vec3 &rhs)
    {
        NOT_IMPLEMENTED
    }

    friend Vec3 operator-(const Vec3 &lhs, const Vec3 &rhs)
    {
        NOT_IMPLEMENTED
    }

    friend Vec3 operator*(const Vec3 &lhs, const Vec3 &rhs)
    {
        NOT_IMPLEMENTED
    }

    friend Vec3 __dot__(const Vec3 &lhs, const Vec3 &rhs)
    {
        NOT_IMPLEMENTED
    }

    friend Vec3 __cross__(const Vec3 &lhs, const Vec3 &rhs)
    {
        NOT_IMPLEMENTED
    }

    friend Vec3 __mul__(const Vec3 &lhs, const Vec3 &rhs)
    {
        NOT_IMPLEMENTED
    }

    // Unary operators:
    Vec3 operator-() {
        NOT_IMPLEMENTED
    }
};

} // math
} // base
} // svso

#endif //SEMANTIC_RELOCALIZATION_VEC3_H
