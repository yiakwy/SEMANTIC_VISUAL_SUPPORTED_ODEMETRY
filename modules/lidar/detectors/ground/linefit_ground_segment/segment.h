//
// Created by yiak on 2021/7/5.
//  Adapted from original codebase https://github.com/lorenwel/linefit_ground_segmentation
//  to replace PCL SACSegmentation algorithm
// Credits to original author
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_SEGMENT_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_SEGMENT_H

#include <memory>
#include <list>
#include <map>

#include "bin.h"

namespace svso {
namespace lidar {
namespace perception {

// we prefer use Sector instead of Segment in program while keep original name style from Linefit Ground Segmentation method
class Segment {
public:
    using Line = std::pair<Bin::MinZPoint, Bin::MinZPoint>;
    // z = a1 * d + a0 for polar line
    using Coefficents = std::pair<double, double>;

    Segment(const unsigned int& n_bins,
            const double& max_slope,
            const double& max_error,
            const double& long_threshold,
            const double& max_long_height,
            const double& max_start_height,
            const double& sensor_height);

    double verticalDistanceToLine(const double& d, const double &z);

    void fitSegmentLines();

    inline Bin& operator[](const size_t& index) {
        return bins_[index];
    }

    inline std::vector<Bin>::iterator begin() {
        return bins_.begin();
    }

    inline std::vector<Bin>::iterator end() {
        return bins_.end();
    }

    bool getLines(std::list<Line>* lines);

private:
    Coefficents fitPoleline(const std::list<Bin::MinZPoint>& points);

    double getMeanError(const std::list<Bin::MinZPoint>& points, const Coefficents& line);

    double getMaxError(const std::list<Bin::MinZPoint>& points, const Coefficents& line);

    Line getPoleline(const Coefficents& local_line, const std::list<Bin::MinZPoint>& line_points);

private:
    // Parameters. Description in GroundSegmentation.
    const double sensor_height_;
    const double max_slope_;
    const double max_error_;
    const double max_start_height_;
    const double long_threshold_;
    const double max_long_height_;

    std::vector<Bin> bins_;

    std::list<Line> lines_;

};

    } // perception
  } // lidar
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_SEGMENT_H
