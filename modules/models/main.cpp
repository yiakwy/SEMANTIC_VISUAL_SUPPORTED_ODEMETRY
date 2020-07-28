//
// Created by yiakwy on 4/3/20.
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

using namespace svso::system;
using namespace svso::models;
using namespace svso::base::logging;

// Parsing comandline inputs
DEFINE_string(graph_def_name, "test_graph", "Static graph definition file's name");
DEFINE_string(suffix, "pb", "Static graph definition file");
DEFINE_string(model_dir, format(
        "%s/models/coco", env_config::DATA_DIR.c_str()
        ),
        "The path of saved model graph definition");

void Parse_args(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}

using tfe = TensorFlowEngine;
namespace tf = tensorflow;

void Create_test_graph_tensors(TensorFlowEngine::InputTensors* inputs, TensorFlowEngine::OutputTensors* outputs)
{
    if (inputs == nullptr || outputs == nullptr) {
        LOG(FATAL) << "[Main] Wrong Value: inputs and outpus should not be null" << std::endl;
    }
    inputs->clear();
    outputs->clear();

    // see examples from https://github.com/cjweeks/tensorflow-cmake/blob/master/examples/external-project/main.cc
    tfe::Tensor x(tf::DT_FLOAT, tf::TensorShape()), y(tf::DT_FLOAT, tf::TensorShape());
    x.scalar<float>()() = 23.0;
    y.scalar<float>()() = 19.0;

    inputs->emplace_back("x", x);
    inputs->emplace_back("y", y);

}

int main(int argc, const char** argv)
{
    Parse_args(argc, (char**)argv);
    Init_GLog(argc, argv);

    const std::string graph_def = format(
            "%s/%s.%s",
            FLAGS_model_dir.c_str(),
            FLAGS_graph_def_name.c_str(),
            FLAGS_suffix.c_str()
            );

    if (!boost::filesystem::exists(graph_def)) {
        LOG(FATAL) << format("[Main] could not find <%s>!", graph_def.c_str());
    }

    if (FLAGS_suffix == "pb") {
        LOG(INFO) << format("[Main] using TensorFlowEngine as backend to infer from inputs.");
        TensorFlowEngine engine(graph_def);

        // define tenors
        tfe::InputTensors inputs;
        tfe::OutputTensors outputs;

        Create_test_graph_tensors(&inputs, &outputs);

        tfe::FutureType fut = engine.Run(inputs, outputs, {"z"}, {});
        // do something else

        // fetch result
        fut.wait();

        tf::Status status = fut.get();
        if (status.ok()) {
            if (outputs.size() == 0) {
                LOG(INFO) << format("[Main] Found no output through <%s>: %s!", graph_def.c_str(), status.ToString().c_str());
                return -1;
            }
            LOG(INFO) << format("[Main] Success: infer through <%s>!", graph_def.c_str());
            tfe::Tensor output = outputs[0];
            LOG(INFO) << format("[Main] fetch result(3 precision): %.3f!", output.scalar<float>());
        } else {
            LOG(INFO) << format("[Main] Failed to infer through <%s>: %s!", graph_def.c_str(), status.ToString().c_str());
        }

    } else {
        LOG(INFO) << format("[Main] <%s> file is not supported.", FLAGS_suffix.c_str());
        return -1;
    }

    return 0;
}