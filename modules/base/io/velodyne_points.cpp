//
// Created by yiak on 2021/4/23.
//

#include "velodyne_points.h"

namespace svso {
namespace base {
namespace io {
namespace reader {

using namespace fast_ground_segment;

pcl::PointCloud<VelodynePoints::PCLPoint>::Ptr
VelodynePoints::EstimatePlane() {
    Eigen::Vector4f ground_coeff;

    LinefitGroundSegmentInitOptions init_options;
    LinefitGroundSegment line_fit_ground_segmentor(init_options);
    std::vector<int> ground_indice_mask;
    std::vector<int> indices;

    line_fit_ground_segmentor.segment(*structured_points_, &ground_indice_mask);

    pcl::PointCloud<PCLPoint>::Ptr ground_sample(new pcl::PointCloud<PCLPoint>);
    for (size_t i=0; i < structured_points_->size(); ++i) {
        if (ground_indice_mask[i] == 1) {
            ground_sample->push_back( (*cloud_)[i] );
            indices.push_back(i);
        }
    }

    auto rnd = [=, &ground_sample]() {
        return ((*rng_gen_)()) % ground_sample->size();
    };

    // Estimate plane parameters
    Point3D p[4];
    Eigen::Vector3f p0p1;
    Eigen::Vector3f p0p2;
    do {
        p[0] = structured_points_->at(indices[rnd()]);
        p[1] = structured_points_->at(indices[rnd()]);
        p[2] = structured_points_->at(indices[rnd()]);
        p[3] = structured_points_->at(indices[rnd()]);

        p0p1 = p[1].vec3f - p[0].vec3f;
        p0p2 = p[2].vec3f - p[0].vec3f;

        // check collinearity
        Eigen::Vector3f ratios = p0p1.array() / (p0p2.array() + 1e-3);
        if (ratios[0] == ratios[1] && ratios[1] == ratios[2]) {
            // bad samples, resample
            continue;
        }
    } while(false);

    Eigen::Vector3f n = p0p1.cross(p0p2);
    n.normalize();
    ground_coeff[0] = n[0];
    ground_coeff[1] = n[1];
    ground_coeff[2] = n[2];
    ground_coeff[3] = -1.0f * (n.dot(p[0].vec3f));

    coefficients_groud_plane_.reset(new pcl::ModelCoefficients);
    coefficients_groud_plane_->values.resize(4);
    memcpy(&coefficients_groud_plane_->values[0], &ground_coeff[0], ground_coeff.size() * sizeof(float));

    return ground_sample;
}

pcl::PointCloud<VelodynePoints::PCLPoint>::Ptr
VelodynePoints::EstimatePlane(pcl::ModelCoefficients::Ptr& coefficients) {
    pcl::PointCloud<PCLPoint>::Ptr ground_sample = EstimatePlane();
    // update passed values
    coefficients->values.resize(coefficients_groud_plane_->values.size());
    memcpy(&coefficients->values[0], &coefficients_groud_plane_->values[0], coefficients_groud_plane_->values.size() * sizeof(float));
    return ground_sample;
}

} // reader
} // io
} // base
} // svso