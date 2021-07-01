//
// Created by yiak on 2020/7/22.
//

#include <gflags/gflags.h>

#include <memory>
#include <string>
#include <unordered_map>
using std::pair;
using std::unordered_map;
#include <vector>
using std::vector;
// if is linux system
#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <stdlib.h>
#include <sstream>

// == Boost Utilities

// filesystem
#include <boost/filesystem.hpp>

// string and STL algorithms
#include <boost/algorithm/string.hpp>
#include <boost/optional.hpp>

// logging utilities
#include <glog/logging.h>
#include "base/logging.h"

#include <iostream>

// TensorFlow Inference Engine
#include "engine.h"
#include "env_config.h"

// SFE
#include "sfe.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "base/parallel_tasking_sched/task.h"
#include "base/io/img.h"

using namespace svso::system;
using namespace svso::models;
using namespace svso::base::logging;
using namespace svso::base::pts;
using namespace svso::base::io;

// Parsing comandline inputs
DEFINE_string(image_name, format(
        "%s/tum/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png", env_config::DATA_DIR.c_str()
),
        "image name");
DEFINE_string(model_dir, format(
        "%s/models/coco", env_config::DATA_DIR.c_str()
),
              "The path of saved model graph definition");

void Parse_args(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}

using tfe = TensorFlowEngine;
namespace tf = tensorflow;

int main(int argc, const char** argv) {
    Parse_args(argc, (char**)argv);
    Init_GLog(argc, argv);

    ThreadPool pool(3);

    // read img, note cv::imread is too slow when you are digesting a bunch of images (> 1TB)
    /* cv::Mat img = cv::imread(FLAGS_image_name, cv::IMREAD_COLOR); */
    Task<std::function<cv::Mat()>>::Ptr read_img(
            new Task<std::function<cv::Mat()>>( [&] {
                reader::RawImageConstPtr raw_image_data = reader::open_with_mmap(FLAGS_image_name); // read the large block of data using mmap, it will take few milli seconds to build mapping
                reader::Img img;
                img.set_data(raw_image_data->data());
                cv::Mat ret = img.mat<int>();
                return ret;
            } )
            );

    auto read_img_fut = read_img->get_future();
    // execute the task and do something else immediately
    pool.enqueue([&] {
        (*read_img)();
    });

    TF_MRCNN_SemanticFeatureExtractor::Ptr sfe(new TF_MRCNN_SemanticFeatureExtractor);

    // wait for the image is ready
    read_img_fut.wait();
    cv::Mat img = std::move( read_img_fut.get() );

    if (img.empty()) {
        LOG(FATAL) << format("could not read the image %s", FLAGS_image_name.c_str());
    }

    tfe::InputTensors inputs;
    tfe::OutputTensors outputs;
    std::vector<TF_MRCNN_SemanticFeatureExtractor::DetectronResult> results;

    // infer !
    tfe::FutureType fut = sfe->detect(img, &inputs, &outputs, results);

    // do something else

    // fetch result
    fut.wait();

    tf::Status status = fut.get();
    if (status.ok()) {
        LOG(INFO) << format("%d objects detected.", results[0].rois.rows());
    }
    return 0;
}