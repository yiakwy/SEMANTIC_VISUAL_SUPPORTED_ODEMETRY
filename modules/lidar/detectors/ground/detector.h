//
// Created by yiak on 2021/4/29.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_DETECTOR_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_DETECTOR_H

#include <memory>

#include <base/exceptions.h>

namespace svso {
namespace lidar {
namespace perception {

class GroundEstimate;
class GroundEstimateImpl;

// @todo TODO, see ${lidar_brand}.proto and ${lidar_brand}_points.h/cpp for details
struct GroundEstimateInitOptions {

};

// integrated to main line perception module
class GroundEstimate {
public:
    using Type = GroundEstimate;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;
    using Impl = GroundEstimateImpl;
    using Impl_ptr = std::shared_ptr<Impl>;

    GroundEstimate() {}
    virtual ~GroundEstimate() { Clear(); }

    // @todo TODO
    void Clear() {}

protected:
    Impl_ptr impl_;
};

// interface for all ground segmentation algorithms
class GroundEstimateImpl {
public:
    using Type = GroundEstimate;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    GroundEstimateImpl() {}
    virtual ~GroundEstimateImpl() { Clear(); }

    // @todo TODO
    void Clear() {}
};

    } // perception
  } // lidar
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_DETECTOR_H
