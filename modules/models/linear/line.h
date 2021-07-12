//
// Created by yiak on 2021/7/12.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_LINE_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_LINE_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <cmath>

#include <sstream>
#include <string>

// LOGGING
#include <glog/logging.h>
#include "base/logging.h"
#include "base/exceptions.h"

#include "env_config.h"

namespace svso {
namespace models {
namespace linear {

using namespace svso::base;
using namespace svso::base::logging;
using namespace svso::execptions;

class LinearModel {
public:
    explicit LinearModel() : is_fitted_(false), offsetY_(0.0) {}

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

    void is_fitted() {
        return is_fitted_;
    }

    CallData& calldata() {
        return calldata_;
    }

    ResStat& stat() {
        return stat_;
    }

    std::string Name() {
        return std::string("Linear Model");
    }
protected:

    class ResStat {
    public:
        /**
         * @desc unbiased standared deviation
         */
        double ustd = 0.0;

        /**
         * @desc first order statistics
         */
        double mean = 0.0;

        /**
         * @desc first order statistics
         */
        double median = 0.0;

        /**
         * @desc quantile ratio upper buond
         */
        double upper_quantile_thred = 0.95;

        /**
         * @desc quantile ratio lower bound
         */
        double lower_quantile_thred = 0.05;

        /**
         * @desc quantile index upper buond
         */
        int upper_quantile_idx = 0.95;

        /**
         * @desc quantile index lower bound
         */
        int lower_quantile_idx = 0.05;

        /**
         * @desc quantile upper bound
         */
        double upper_quantile = 0.0;

        /**
         * @desc quantile lower bound
         */
        double lower_quantile = 0.0;

        /**
         * @desc input data
         */
        std::vector<double> array;

        ResStat() {}
        ResStat(std::vector<double>& _array) : array(_array) {
            double sum = 0.0;
            size_t l = array.size();
            lower_quantile_idx = lower_quantile_thred * l;
            upper_quantile_idx = upper_quantile_thred * l;
            double median_idx = 0.5 * l;
            if (lower_quantile_idx <= 0) {
                lower_quantile_idx = 1;
            }

            if (upper_quantile_idx >= l-1) {
                upper_quantile_idx = l-2;
            }

            // computing quantile
            std::vector<double> temp(array);
            std::sort(temp.begin(), temp.end());
            lower_quantile = temp[lower_quantile_idx];
            upper_quantile = temp[upper_quantile_idx];
            median         = temp[median_idx];

            // computing first order statistics
            for (int i=0; i < l; i++) {
                sum += array[i];
            }
            mean = sum / (l-1.f);
            sum = 0.0;
            for (int i=0; i < l; i++) {
                sum += pow(array[i] - mean, 2);
            }
            ustd = sum / (l-1.f);
            ustd = sqrt(ustd);
        }
    };

    class CallData {
    public:
        Eigen::VectorXd coefficients;
        double intercept;

    };

    CallData calldata_;
    Eigen::VectorXd offsetX_;
    double offsetY_;

    // residual statistics
    ResStat stat_;

    // training parameters
    bool normalize_ = true;

    // meta data
    bool is_fitted_ = false;
};


    } // linear
  } // models
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_LINE_H
