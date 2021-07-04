//
// Created by yiak on 2021/4/23.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_VELODYNE_POINTS_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_VELODYNE_POINTS_H

#include <iostream>
#include <memory>
using std::static_pointer_cast;

#include <vector>
using std::vector;

#include <set>
using std::set;

#include <string>

// used for RangeImage representation
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
// some useful ISP libraries
#include <opencv2/imgproc.hpp>

#ifndef NDEBUG
#include <opencv2/highgui.hpp>
#endif

#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

// available for pcl-1.11 but not for pcl-1.8
// #include <pcl/pcl_memory.h>
// #include <pcl/pcl_macros.h>

#include <pcl/pcl_base.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
// #include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/ModelCoefficients.h>

#include <math.h>
#include <cmath>

#include <ctime>
#include <climits>

#include <thread>

// parallel container
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_hash_map.h>

// math
#include <base/math/vec3.h>
#include <base/array_like_hasher.h>

// logging
#include <base/logging.h>
#include <base/exceptions.h>

#include "env_config.h"

#include "devices/velodyne_points.pb.h"
#include "base/io/sensors/flags/velodyne_points_flags.h"
#include "base/flags/all_flags.h"

namespace svso {
namespace base {
namespace io {
namespace reader {

using namespace svso::system;
using namespace svso::base::logging;
using namespace svso::base::exceptions;
using namespace svso::common::io::sensors;
using namespace svso::base::math;
namespace fs = boost::filesystem;

class Point3D : public Vec3 {
public:
    using Base = Vec3;
    using Type = Point3D;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    Point3D () : Vec3(),
                 rowId(-1),
                 colId(-1),
                 d(0),
                 intensity(0) {}
    explicit Point3D(double x, double y, double z=0) : Vec3(x, y, z) {}
    virtual ~Point3D () {}

    using VoxelKey = int64_t;

    /*
     * attributes
     */
    union {
        int size[4];
        struct {
            int rowId;
            int colId;
            // Used in fitting a line a1*d + a0 = z originating from (0,0) in local sensor coordinate system.
            // Note for a line (d, z, theta), theta is always constant.
            float d;
            float intensity;
        };
    };

    /* int64_t is not allowed in PCL primitive point field */
    VoxelKey key;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Complementary to ::Point3D
class PointMeta {
public:
    PointMeta() {}
    virtual ~PointMeta() {}

    using GridKey = int64_t;

    /*
     * meta attributes
     */
    size_t point_id;

    union {
        // additional two fields for padding
        int size[4];
        struct {
            int row_id;
            int col_id;
            // Used in fitting a line a1*d + a0 = z originating from (0,0) in local sensor coordinate system.
            // Note for a line (d, z, theta), theta is always constant.
            float d;
            float intensity;
        };
    };

    // used for curvature feature extraction
    int ring_id;

    // used for motion compensation
    double timestamp;

    GridKey voxel_grid_key;
    GridKey occpc_grid_key;

    enum class PointType {
             UNDEF = 0,
        FOREGROUND = 1,
        BACKGROUND = 2,
       SMALL_MOTOR = 3,
         BIG_MOTOR = 4,
        PEDESTRAIN = 5
    };
    PointType type;

};

struct VelodynePointsInitOptions {
    // series type
    string series_type = "VLP16";

    int lines = 16;
    float horizontal_res = 0.2;
    float vertical_res = 2.0;
    float detect_depth = 280;
    float horizontal_span_of_view = 360;
    float vertical_span_of_view = 30;
    float lowest_angle = 15.0+0.1;
    // our velodyne is configured with additional angles
    float device_angle = 3;
    float minimum_range = 1.0;
    int threads = -1;
};

struct InnoPointsInitOptions {
    // series type
    string series_type = "RawJaguar100";

    // see product official website http://www.innovusion.com/product_show.php?id=392
    int lines = 300;
    float horizontal_res = 0.1; // after negotiating with innovusion developers, resolusion values can be changed to smaller ones
    float vertical_res = 0.1;
    float detect_depth = 280;
    float horizontal_span_of_view = 100;
    float vertical_span_of_view = 40;
    float lowest_angle = 20; // Jaguar100 +-20 degree field of view in vertical direction
    // our Jaguar is configured with additional angles
    float device_angle = 3;
    float minimum_range = 1.0;
    int threads = -1;
};

struct LivoxPointsInitOptions {
    // series type
    string series_type = "Horizon|Tel";

    // see product official website https://www.livoxtech.com/horizon
    int lines = 5;
    float horizontal_res = 0.08;//0.03;
    float vertical_res = 0.28;
    float detect_depth = 1000; // Tel 1000, Horizon 260
    float horizontal_span_of_view = 2*81.7;
    float vertical_span_of_view = 25.1;
    float lowest_angle = 12.55; // Horizon +-12.55 degree field of view in vertical direction
    // our Jaguar is configured with additional angles
    float device_angle = -5; // installation angle
    float minimum_range = 1.0;
    int threads = -1;
};

class VelodynePoints {
public:
    using Type = VelodynePoints;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    VelodynePoints() {}

    virtual ~VelodynePoints() { Clear(); }

    void Clear() {}

    using PCLPoint = pcl::PointXYZI;
    using InitOptions = VelodynePointsInitOptions;

    // used for upgrading to LidarFrame
    using PointKey = size_t;
    using PointMetaBucket = tbb::concurrent_hash_map<PointKey, PointMeta>;

    using RangeKey = std::pair<int, int>;
    /* used for inverse query table */
    using RangeBucket = tbb::concurrent_hash_map<RangeKey, std::set<PointKey>, base::_pair_hasher<RangeKey>>;

    void Init(InitOptions& options) {
        init_options_ = options;

        int vertical_scans = init_options_.vertical_span_of_view / init_options_.vertical_res;
        int horizontal_scans = init_options_.horizontal_span_of_view / init_options_.horizontal_res;
        range_image_ = cv::Mat(vertical_scans, horizontal_scans, CV_32F, cv::Scalar::all(FLT_MAX));

        int n_threads = options.threads;
        if (n_threads < 1) {
            n_threads = std::thread::hardware_concurrency() - 1;
        }

        sched_.reset(new tbb::task_scheduler_init(n_threads));
    }

    /*
     * compute structurs of the point cloud
     * ref:
     *   LeGO LOAM: follow the main logics defined in LeGO-LOAM/src/imageProjection.cpp, line 211~257
     */
    void init_structure() {
        // range image projection
        size_t num_returns = cloud_->points.size();

        int vertical_scans = range_image_.rows;
        int horizontal_scans = range_image_.cols;

        structured_points_.reset(new pcl::PointCloud<Point3D>);
        structured_points_->resize(num_returns);
        LOG(INFO) << format("[VelodynePoints::_init_structure] Original velodyne points returns number %zu", num_returns);

        // estimate load factor of the buckets
        point_meta_bucket_.clear();
        range_bucket_.clear();

        float point_meta_bucket_load_factor = num_returns / (point_meta_bucket_.bucket_count() + 1e-3);
        float range_bucket_load_factor = num_returns / (range_bucket_.bucket_count() + 1e-3);

        if (point_meta_bucket_load_factor > 0.5) {
            size_t estimated = (num_returns + point_meta_bucket_.bucket_count()) * 2;
            // reallocating estimated slots
            point_meta_bucket_.rehash(estimated);
        }

        if (range_bucket_load_factor > 0.5) {
            size_t estimated = (num_returns + range_bucket_.bucket_count()) * 2;
            // reallocating estimated slots
            range_bucket_.rehash(estimated);
        }

        // cpu version using block range format
        auto cpu_kernel = [=, this] (const tbb::blocked_range<size_t>& range) {

            float verticalAngle, horizonAngle, rangeVal;
            size_t rowIdx, columnIdx, index;
            Point3D thisPoint;

            for (size_t i=range.begin(); i != range.end(); ++i) {
                thisPoint.x = cloud_->points[i].x;
                thisPoint.y = cloud_->points[i].y;
                thisPoint.z = cloud_->points[i].z;
                thisPoint.intensity = cloud_->points[i].intensity;

                // build (d,z) grid for fast ground segmentation by estimating lines
                thisPoint.d = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y);

                verticalAngle = atan2(thisPoint.z, thisPoint.d) * 180 / M_PI;
                rowIdx = vertical_scans - round((verticalAngle + init_options_.lowest_angle + init_options_.device_angle) / init_options_.vertical_res);
                if (rowIdx < 0 || rowIdx > vertical_scans) {
                    continue;
                }

                horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                columnIdx = -round((horizonAngle - 90.0) / init_options_.vertical_res) + horizontal_scans / 2;
                if (columnIdx >= horizontal_scans)
                    columnIdx -= horizontal_scans;

                if (columnIdx < 0 || columnIdx >= horizontal_scans) {
                    continue;
                }

                rangeVal = thisPoint.intensity;
                if (rangeVal < init_options_.minimum_range) {
                    continue;
                }

                thisPoint.rowId = rowIdx;
                thisPoint.colId = columnIdx;
                structured_points_->points[i] = thisPoint;

                /*
                 * write additional info into concurrent containers
                 */
                // point_meta_bucket_
                PointMetaBucket::accessor point_meta_bucket_writer;
                PointMetaBucket::const_accessor point_meta_bucket_reader;

                PointKey point_key = i;
                if (!point_meta_bucket_.find(point_meta_bucket_reader, point_key)) {
                    point_meta_bucket_.insert(point_meta_bucket_writer, point_key);
                    PointMeta point_meta;
                    point_meta.point_id = point_key;
                    point_meta.row_id = rowIdx;
                    point_meta.col_id = columnIdx;
                    point_meta_bucket_writer->second = point_meta;
                    point_meta_bucket_writer.release();
                } else {
                    // handling exception here
                    const PointMeta& point_meta = point_meta_bucket_reader->second;
                    if (point_meta.row_id != rowIdx || point_meta.col_id != columnIdx) {
                        LOG(FATAL) << "Wrong Duplicated Value!";
                    }
                }

                // range_bucket_
                RangeBucket::accessor range_bucket_writer;
                RangeBucket::const_accessor range_bucket_reader;

                RangeKey range_key(rowIdx, columnIdx);
                if (!range_bucket_.find(range_bucket_writer, range_key)) {
                    range_bucket_.insert(range_bucket_writer, range_key);
                    std::set<PointKey> keys;
                    keys.insert(point_key);
                    range_image_.at<float>(rowIdx, columnIdx) = rangeVal;
                    range_bucket_writer->second = keys;

                } else {

                    range_bucket_writer->second.insert(point_key);
                    int size = range_bucket_writer->second.size();
                    // use average rangeVal
                    range_image_.at<float>(rowIdx, columnIdx) =  (range_image_.at<float>(rowIdx, columnIdx) * (size-1)+ rangeVal) / (1.f * size);
                }
                range_bucket_writer.release();

            }

        };

        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_returns), cpu_kernel);
        structured_points_->width = num_returns;
        structured_points_->height = 1;
        structured_points_->is_dense = true;

        // transform the range values probability distribution
        // The operation is expensive in local machine
        if (FLAGS_use_ostu_light_equalization) {
            range_image_.convertTo(range_image_, CV_8UC1);
            cv::equalizeHist(range_image_, range_image_);
        }

        // for test purpose
        if (FLAGS_all_debug) {
            std::string log_dir = format("%s/log/range_images", env_config::LOG_DIR.c_str());;
            fs::path path(log_dir);
            if (!fs::exists(path)) {
                LOG(WARNING) << format("Directory %s does not exist", log_dir.c_str());
                fs::create_directories(path);
            }

            std::string filename = format("%s/frame_%d.png", log_dir.c_str(), frame_id_);
            cv::imwrite(filename.c_str(), range_image_);
        }
    }

    /*
    * protected/private attributes accessor
    */
    const std::string &data() const { return data_; }

    const std::string &set_data(const std::string &data) {
        data_ = data;
        return data_;
    }

    void set_frame_id(int frame_id) {
        frame_id_ = frame_id;
    }

    int frame_id() {
        return frame_id_;
    }

    void set_cloud(pcl::PointCloud<PCLPoint>::Ptr& cloud) {
        cloud_ = cloud;
    }

    const pcl::PointCloud<PCLPoint>::Ptr& cloud() {
        return cloud_;
    }

    // fetch computed structural point cloud
    const pcl::PointCloud<Point3D>::Ptr& structured_points() {
        return structured_points_;
    }

    cv::Mat range_image() {
        return range_image_;
    }

    PointMetaBucket point_meta_bucket() {
        return point_meta_bucket_;
    }

    RangeBucket range_bucket() {
        return range_bucket_;
    }

private:
    std::string data_;

    // suppose the data has already been decoded with PCD v7 format decoder
    pcl::PointCloud<PCLPoint>::Ptr cloud_;
    int frame_id_;

    // keep a copy of the original point cloud
    pcl::PointCloud<Point3D>::Ptr structured_points_;

    // estimated structure parameters for the point cloud
    VelodynePointsInitOptions init_options_;

    cv::Mat range_image_;
private:
    PointMetaBucket point_meta_bucket_;
    RangeBucket range_bucket_;
    std::shared_ptr<tbb::task_scheduler_init> sched_;
};

} // reader

// @todo : TODO
namespace writer {} // writer

    } // io
  } // base
} // svso

// register the point type to PCL codebase to enable the use of PCL algorithms
POINT_CLOUD_REGISTER_POINT_STRUCT(svso::base::io::reader::Point3D,
                           (float, x, x)
                                 (float, y, y)
                                 (float, z, z)
                                 (int, rowId, rowId)
                                 (int, colId, colId)
                                 (float, d, d)
                                 (float, intensity, intensity))

#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_VELODYNE_POINTS_H
