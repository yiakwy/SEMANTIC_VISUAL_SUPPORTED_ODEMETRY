//
// Created by yiak on 2021/7/5.
//

#include <chrono>
#include <cmath>
#include <list>
#include <memory>
#include <thread>

// parallel container and optimized with threads pool
#include "tbb/task_scheduler_init.h"
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"
#include <tbb/parallel_for.h>

#include "base/logging.h"

#include "linefit_ground_segment.h"

namespace svso {
namespace lidar {
namespace perception {

LinefitGroundSegment::LinefitGroundSegment(const LinefitGroundSegmentInitOptions &init_options) :
        init_options_(init_options),
        segments_(init_options.n_segments, Segment(init_options.n_bins,
                                                   init_options.max_slope,
                                                   init_options.max_error_square,
                                                   init_options.long_threshold,
                                                   init_options.max_long_height,
                                                   init_options.max_start_height,
                                                   init_options.sensor_height)) {}

void LinefitGroundSegment::segment(const PointCloud &cloud, std::vector<int> *segmentation) {
    segmentation->clear();
    segmentation->resize(cloud.size(), 0);

    // build (d, z) grid
    insertPoints(cloud);

    fitLines();

    assignCluster(segmentation);
}

void LinefitGroundSegment::insertPoints(const PointCloud &cloud) {

    bin_index_.resize(cloud.size());
    segment_coordinates_.resize(cloud.size());

    auto cpu_kernel = [=, &cloud] (const tbb::blocked_range<size_t>& range) {

        float verticalAngle, horizonAngle, rangeVal;
        int sector_idx, range_idx;

        const double sector_res = 360.0 / init_options_.n_segments;
        const double range_step_res = (init_options_.r_max - init_options_.r_min) / init_options_.n_bins;

        int horizontal_scans = init_options_.sensor_points_init_options.horizontal_span_of_view /
                               sector_res;

        for (size_t i = range.begin(); i < range.end(); ++i) {
            PCLPoint point(cloud[i]);
            /*
            double d = point.d;
            */
            double d = sqrt(point.x * point.x + point.y * point.y);
            if (d < init_options_.r_max && d > init_options_.r_min) {
                /*
                sector_idx = point.colId;
                */
                /*
                horizonAngle = std::atan2(point.x, point.y) * 180 / M_PI;
                sector_idx = -round((horizonAngle-90) / sector_res) + horizontal_scans / 2;
                */
                horizonAngle = std::atan2(point.y, point.x) * 180 / M_PI;
                sector_idx = (horizonAngle + 180) / sector_res;
                if (sector_idx >= init_options_.n_segments) {
                    sector_idx -= init_options_.n_segments;
                }

                if (sector_idx < 0 || sector_idx >= init_options_.n_segments) {
                    bin_index_[i] = std::make_pair<int, int>(-1, -1);
                    segment_coordinates_[i] = Bin::MinZPoint(d, point.z);
                    continue;
                }

                range_idx = (d - init_options_.r_min) / range_step_res;

                segments_[sector_idx][range_idx].addPoint(d, point.z);
                bin_index_[i] = std::make_pair(sector_idx, range_idx);
            } else {
                bin_index_[i] = std::make_pair<int, int>(-1, -1);
            }
            segment_coordinates_[i] = Bin::MinZPoint(d, point.z);

        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, cloud.size()), cpu_kernel);

}

void LinefitGroundSegment::fitLines() {
    {
        const size_t parallelism = tbb::task_scheduler_init::default_num_threads();
        tbb::global_control threads_control(tbb::global_control::max_allowed_parallelism, init_options_.n_threads_used_for_linefit);

        auto cpu_kernel =
                [=](const tbb::blocked_range<size_t> &range) {

                    for (size_t i=range.begin(); i != range.end(); ++i) {
                        segments_[i].fitSegmentLines();
                    }

                };
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, init_options_.n_segments),
                cpu_kernel);
    }

}

void LinefitGroundSegment::assignCluster(std::vector<int> *segmentation) {
    auto cpu_kernel = [=, &segmentation] (const tbb::blocked_range<size_t>& range) {

        float verticalAngle, horizonAngle, rangeVal;
        int sector_idx, range_idx;

        const double sector_res = 360.0 / init_options_.n_segments;
        const double range_step_res = (init_options_.r_max - init_options_.r_min) / init_options_.n_bins;

        int horizontal_scans = init_options_.sensor_points_init_options.horizontal_span_of_view /
                               sector_res;

        for (size_t i=range.begin(); i < range.end(); i++) {
            Bin::MinZPoint min_z_point = segment_coordinates_[i];
            sector_idx = bin_index_[i].first;
            if (sector_idx >= 0) {
                double dist = segments_[sector_idx].verticalDistanceToLine(min_z_point.d, min_z_point.z);
                int steps = 1;
                while (dist == -1 && steps * sector_res * M_PI / 180 < init_options_.line_search_angle) {

                    int end_idx = sector_idx + steps;
                    while (end_idx > init_options_.n_segments) {
                        end_idx -= init_options_.n_segments;
                    }
                    int start_idx = sector_idx - steps;
                    while (start_idx < 0) {
                        start_idx += init_options_.n_segments;
                    }

                    double d1 = segments_[end_idx].verticalDistanceToLine(min_z_point.d, min_z_point.z);
                    double d2 = segments_[start_idx].verticalDistanceToLine(min_z_point.d, min_z_point.z);

                    if (d1 > dist) {
                        dist = d1;
                    }

                    if (d2 > dist) {
                        dist = d2;
                    }

                    ++steps;

                } // end while

                if (dist < init_options_.max_dist_to_line && dist != -1) {
                    segmentation->at(i) = 1;
                }

            } // end if

        } // end for
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, segmentation->size()), cpu_kernel);

}

    } // perception
  } // lidar
} // svso

