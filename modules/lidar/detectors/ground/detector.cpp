//
// Created by yiak on 2021/4/29.
//

#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>

#include "detector.h"

namespace svso {
namespace lidar {
namespace perception {

void SequentialSACLinefitGroundEstimateImpl::EstimatePlane(const pcl::PointCloud<PCLPoint>& structured_points) {
    // copy the header information
    inliers_->header = coefficients_->header = structured_points.header;
    input_ = (pcl::PointCloud<PCLPoint>*)(&structured_points);

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
        sqrt_dist_error_ = std::numeric_limits<double>::max() - 1;
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
    double last_sqrt_dist_error = sqrt_dist_error_;

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

    // linefit estimator
    std::vector<int> pole_line_indices_mask;
    std::vector<int> pole_line_indices;

    LinefitGroundSegmentInitOptions init_options;
    LinefitGroundSegment line_fit_ground_segmentor(init_options);

    line_fit_ground_segmentor.segment(structured_points, &pole_line_indices_mask);

    pcl::PointCloud<PCLPoint>::Ptr ground_sample(new pcl::PointCloud<PCLPoint>);
    for (size_t i=0; i < structured_points.size(); ++i) {
        if (pole_line_indices_mask[i] == 1) {
            ground_sample->push_back( structured_points[i] );
            pole_line_indices.push_back(i);
            inliers_->indices.push_back(i);
        }
    }

    // main loop
    while (i < MaxIters-1 && skipped_count < max_skip) {
        i++;

        if (i==0) {
            double sqrt_dist_err = selectWithinDistance(refined_model_coefficients, thresh_, inliers_->indices);
            for (size_t j=0; j < sample_size; j++) {
                best_selection.push_back(inliers_->indices[j]);
            }
            samples = best_selection;
            n_best_inliers_count = inliers_->indices.size();

            update_k_iter();

            if (max_skip > k) {
                // using our guessed iterations
                max_skip = k;
                LOG(INFO) << "[SequentialSACLinefitGroundEstimateImpl::EstimatePlane] guessed iterations are used for skipping : " << max_skip;
            }
        } else {
            // perform refinement and use heuristic samples instead of random samples
            getHeuristicSamples(i, samples, pole_line_indices);
        }
        if (samples.empty()) {
            LOG(ERROR) << format("[SequentialSACLinefitGroundEstimateImpl::EstimatePlane] No samples could be selected!\n");
            break;
        }

        if (!computeModelCoefficients(samples, coeff, structured_points)) {
            ++skipped_count;
            LOG(ERROR) << format("[SequentialSACLinefitGroundEstimateImpl::EstimatePlane] No coefficients could be estimated!\n");
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
SequentialSACLinefitGroundEstimateImpl::getHeuristicSamples(int i, std::vector<int>& samples, const std::vector<int>& estimates) {
    size_t size = estimates.size();
    samples.clear();

    auto rnd = [=]() {
        return ((*rng_gen_)()) % size;
    };

    for (int i=0; i < 4; i++) {
        samples.push_back(estimates[rnd()]);
    }
}

double
SequentialSACLinefitGroundEstimateImpl::selectWithinDistance(const Eigen::VectorXf& coefficients, double thresh, vector<int>& indices) {
    size_t size = (*input_).size();

    indices.clear();
    indices.resize(size);

    size_t j=0;
    double sqrt_dist_error = 0.0;
    for (size_t i=0; i < size; i++) {
        auto pt = (*input_).points[i];
        Eigen::Vector4f h_p;
        h_p << pt.x, pt.y, pt.z, 1;
        float dist_to_plane = fabsf(coefficients.dot(h_p));
        if (dist_to_plane < thresh) {
            indices[j] = i;
            sqrt_dist_error += dist_to_plane;
            j++;
        }
    }

    CHECK(size > 0);
    sqrt_dist_error /= size;
    return sqrt_dist_error;
}

bool
SequentialSACLinefitGroundEstimateImpl::computeModelCoefficients(const std::vector<int>& samples, Eigen::VectorXf& coeff, const pcl::PointCloud<PCLPoint>& structured_points) {
    Point3D p[3];
    Eigen::Vector3f p0p1;
    Eigen::Vector3f p0p2;

    CHECK(samples.size() >= 4 );

    p[0] = structured_points.at(samples[0]);
    p[1] = structured_points.at(samples[1]);
    p[2] = structured_points.at(samples[2]);

    p0p1 = p[1].vec3f - p[0].vec3f;
    p0p2 = p[2].vec3f - p[0].vec3f;

    // check collinearity
    Eigen::Vector3f ratios = p0p1.array() / (p0p2.array() + 1e-3);
    if (ratios[0] == ratios[1] && ratios[1] == ratios[2]) {
        // bad samples, resample
        return false;
    }

    Eigen::Vector3f n = p0p1.cross(p0p2);
    if (n[2] < 0) {
        n = n * -1;
    }
    n.normalize();
    coeff[0] = n[0];
    coeff[1] = n[1];
    coeff[2] = n[2];
    coeff[3] = -1.0f * (n.dot(p[0].vec3f));
    return true;
}

size_t
SequentialSACLinefitGroundEstimateImpl::countWithinDistance(const Eigen::VectorXf& coeff, double thresh) {
    size_t size = (*input_).size();

    size_t j=0;

    for (size_t i=0; i < size; i++) {
        auto pt = (*input_).points[i];
        Eigen::Vector4f h_p;
        h_p << pt.x, pt.y, pt.z, 1;
        float dist_to_plane = fabsf(coeff.dot(h_p));
        if (dist_to_plane < thresh) {
            j++;
        }
    }

    return j;
}

void
SequentialSACLinefitGroundEstimateImpl::optimizeModelCoefficients(vector<int>& indices, const Eigen::VectorXf& coeff, Eigen::VectorXf& coeff_refined) {
    // Needs a valid set of model coefficients
    if (coeff.size () != 4)
    {
        LOG(ERROR) << format("[SequentialSACLinefitGroundEstimateImpl::optimizeModelCoefficients] Invalid number of model coefficients given (%lu)!\n", coeff.size ());
        coeff_refined = coeff;
        return;
    }

    // Need more than the minimum sample size to make a difference
    if (indices.size() < 3)
    {
        LOG(ERROR) << format("[SequentialSACLinefitGroundEstimateImpl::optimizeModelCoefficients] Not enough inliers found to optimize model coefficients (%lu)! Returning the same coefficients.\n", indices.size());
        coeff_refined = coeff;
        return;
    }

    Eigen::Vector4f plane_parameters;

    // Use Least-Squares to fit the plane through all the given sample points and find out its coefficients
    EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
    Eigen::Vector4f xyz_centroid;

    pcl::computeMeanAndCovarianceMatrix (*input_, indices, covariance_matrix, xyz_centroid);

    // Compute the model coefficients
    EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_value;
    EIGEN_ALIGN16 Eigen::Vector3f eigen_vector;
    pcl::eigen33 (covariance_matrix, eigen_value, eigen_vector);

    // Hessian form (D = nc . p_plane (centroid here) + p)
    coeff_refined.resize (4);
    coeff_refined[0] = eigen_vector [0];
    coeff_refined[1] = eigen_vector [1];
    coeff_refined[2] = eigen_vector [2];
    coeff_refined[3] = 0;
    coeff_refined[3] = -1 * coeff_refined.dot (xyz_centroid);
}

    } // perception
  } // lidar
} // svso