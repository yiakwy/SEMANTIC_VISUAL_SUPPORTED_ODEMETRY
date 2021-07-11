//
// Created by yiak on 2021/4/29.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_DETECTOR_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_DETECTOR_H

#include <memory>

#include "base/timmer.h"
#include "base/logging.h"
#include "base/exceptions.h"
#include "base/klass_registry.h"

#include "base/io/sensors/velodyne_points.h"
#include "flags/sequential_linefit_flags.h"


namespace svso {
namespace lidar {
namespace perception {

using namespace svso::base;
using namespace svso::base::timmer;
using namespace svso::base::logging;

class GroundEstimate;

// @todo TODO, see ${lidar_brand}.proto and ${lidar_brand}_points.h/cpp for details
struct GroundEstimateInitOptions {

};


// interface for all ground segmentation algorithms
class GroundEstimate {
public:
    using Type = GroundEstimate;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    GroundEstimate() { Init(); }
    virtual ~GroundEstimate() { Clear(); }

    virtual void Clear() {}

    virtual void Init() {
        inliers_.reset(new pcl::PointIndices);
        coefficients_.reset(new pcl::ModelCoefficients);
    }

    pcl::ModelCoefficients::Ptr coefficients() {
        return coefficients_;
    }

    pcl::PointIndices::Ptr inliers() {
        return inliers_;
    }

protected:
    pcl::PointIndices::Ptr inliers_;
    pcl::ModelCoefficients::Ptr coefficients_;
};


// The algorithm was proposed and implemented based on my original implementation of
//   1. SequentialSACSegment which inherits base class from PCL to modified RANSAC algorithm to reuse estimate
//   2. Linefit algorithm for fast sampling (already shiped with this lidar package)
// Then I quickly realised that I can combine them together: fast sampling randomly!
class SequentialSACLinefitGroundEstimateImpl : public GroundEstimate {
public:

    using PCLPoint = svso::base::io::reader::Point3D;

    SequentialSACLinefitGroundEstimateImpl() : GroundEstimate() { Init(); }
    virtual ~SequentialSACLinefitGroundEstimateImpl() {}

    void Clear() override {}

    void Init() override {
        // setup random sampler
        rng_alg_.seed(static_cast<unsigned int> (std::time(nullptr)));
        rng_dist_.reset (new boost::uniform_int<> (0, std::numeric_limits<int>::max ()));
        rng_gen_.reset (new boost::variate_generator<boost::mt19937&, boost::uniform_int<> > (rng_alg_, *rng_dist_));

        thresh_ = FLAGS_dist_thresh;
        iterations_ = FLAGS_sequential_max_iterations;
    }

    void set_last_guess(pcl::ModelCoefficients::Ptr guess) {
        *last_guess_ = *guess;
    }

    void EstimatePlane(const pcl::PointCloud<PCLPoint>& structured_points);

    void getSamples(int i, std::vector<int>& samples);

    void selectWithinDistance(const Eigen::VectorXf& coefficients, double thresh, vector<int>& indices);

    bool computeModelCoefficients(const std::vector<int>& samples, Eigen::VectorXf& coeff);

    size_t countWithinDistance(const Eigen::VectorXf& coeff, double thresh);

    void optimizeModelCoefficients(vector<int>& indices, const Eigen::VectorXf& coeff, Eigen::VectorXf& coeff_refined);

private:

    // Mersenne Twister based random sampler, see tutorial : https://github.com/yiakwy/yiakwy.github.io/tree/master/Computing%20Random%20Variables
    // this is also the underlying algorithm used inside PCL sampler engine
    boost::mt19937 rng_alg_;
    std::shared_ptr<boost::uniform_int<> > rng_dist_;
    std::shared_ptr<boost::variate_generator< boost::mt19937&, boost::uniform_int<> > > rng_gen_;
    double thresh_;
    int iterations_;
    pcl::ModelCoefficients::Ptr last_guess_;
};

MODEL_REGISTER_SUBCLASS(GroundEstimate, SequentialSACLinefitGroundEstimateImpl);

    } // perception
  } // lidar
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_DETECTOR_H
