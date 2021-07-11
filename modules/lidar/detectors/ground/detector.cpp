//
// Created by yiak on 2021/4/29.
//

#include "detector.h"

namespace svso {
namespace lidar {
namespace perception {

void SequentialSACLinefitGroundEstimateImpl::EstimatePlane(const pcl::PointCloud<PCLPoint>& structured_points) {
    // copy the header information
    inliers_->header = coefficients_->header = structured_points.header;

    inliers_->indices.clear();
    coefficients_->values.clear();

    int MaxIters = iterations_;
    int i = -1;
    size_t n_best_inliers_count = 0;
    size_t sample_size = 3;
    size_t num_points = structured_points.size();
    if (last_guess_.get() == nullptr) {
        LOG(WARNING) << "No valid guess is set.";
        last_guess_.reset(new pcl::ModelCoefficients);
        MaxIters = FLAGS_max_iterations;
    }

    // Indices samples;
    std::vector<int> samples;
    std::vector<int> best_selection;

    // try to improve the current model, no threads is used for the moment
    Eigen::VectorXf refined_model_coefficients;
    Eigen::VectorXf coeff;
    refined_model_coefficients.resize(last_guess_->values.size());

    refined_model_coefficients << last_guess_->values[0],
            last_guess_->values[1],
            last_guess_->values[2],
            last_guess_->values[3];

    // iteration choosing algorithms from PCL Ransac
    double k = 1.0;
    double log_probability  = log (1.0 - 0.99);
    double one_over_indices = 1.0 / static_cast<double> (num_points);

    unsigned skipped_count = 0;
    unsigned max_skip = MaxIters;
    unsigned minimum_iterations = 3;

    auto update_k_iter = [=,&k, &n_best_inliers_count, &log_probability] () {
        // Compute the k parameter (k=log(z)/log(1-w^n))
        double w = static_cast<double> (n_best_inliers_count) * one_over_indices;
        double p_no_outliers = 1.0 - pow (w, static_cast<double> (samples.size ()));
        p_no_outliers = (std::max) (std::numeric_limits<double>::epsilon (), p_no_outliers);       // Avoid division by -Inf
        p_no_outliers = (std::min) (1.0 - std::numeric_limits<double>::epsilon (), p_no_outliers);   // Avoid division by 0.
        k = log_probability / log (p_no_outliers);

        // in case that k is computed as 0, always run additional 3 times
        if (k < minimum_iterations) {
            k = minimum_iterations;
        }
    };

    // main loop
    while (i < MaxIters-1 && skipped_count < max_skip) {
        i++;

        if (i==0) {
            selectWithinDistance(refined_model_coefficients, thresh_, inliers_->indices);
            for (size_t j=0; j < sample_size; j++) {
                best_selection.push_back(inliers_->indices[j]);
            }
            samples = best_selection;
            n_best_inliers_count = inliers_->indices.size();

            update_k_iter();

            if (max_skip > k) {
                // using our guessed iterations
                max_skip = k;
                LOG(INFO) << "[SequentialSACSegment] guessed iterations are used for skipping : " << max_skip;
            }
        } else {
            // perform refinement and use random samples
            getSamples(i, samples);
        }
        if (samples.empty()) {
            LOG(ERROR) << format("[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n");
            break;
        }

        if (!computeModelCoefficients(samples, coeff)) {
            ++skipped_count;
            PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] No coefficients could be estimated!\n");
            continue;
        }

        size_t n_inliers_count = countWithinDistance (coeff, thresh_);
        if (n_inliers_count > n_best_inliers_count) {
            LOG(INFO) << "[SequentialSACSegment] the algorithm is improved at iteration : " << i;
            n_best_inliers_count = n_inliers_count;

            refined_model_coefficients = coeff;
            best_selection = samples;

            update_k_iter();
        }

        if (i >= k) {
            break;
        }
    }

    // update indices
    selectWithinDistance(refined_model_coefficients, thresh_, inliers_->indices);

    // get the model coefficients
    coeff = refined_model_coefficients;

    // optimize the coefficients
    {
        Eigen::VectorXf coeff_refined;
        optimizeModelCoefficients(inliers_->indices, coeff, coeff_refined);
        coefficients_->values.resize(coeff_refined.size());
        memcpy(&coefficients_->values[0], &coeff_refined[0], coeff_refined.size() * sizeof(float));
        // Refine inliers
        selectWithinDistance(coeff_refined, thresh_, inliers_->indices);
    }
}

// implement these methods with the add of  linefit algorithm

void
SequentialSACLinefitGroundEstimateImpl::getSamples(int i, std::vector<int>& samples) {

}

void
SequentialSACLinefitGroundEstimateImpl::selectWithinDistance(const Eigen::VectorXf& coefficients, double thresh, vector<int>& indices) {

}

bool
SequentialSACLinefitGroundEstimateImpl::computeModelCoefficients(const std::vector<int>& samples, Eigen::VectorXf& coeff) {

}

size_t
SequentialSACLinefitGroundEstimateImpl::countWithinDistance(const Eigen::VectorXf& coeff, double thresh) {

}

void
SequentialSACLinefitGroundEstimateImpl::optimizeModelCoefficients(vector<int>& indices, const Eigen::VectorXf& coeff, Eigen::VectorXf& coeff_refined) {

}

    } // perception
  } // lidar
} // svso