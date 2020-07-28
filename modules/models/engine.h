//
// Created by yiakwy on 4/7/20.
//

#pragma once

#ifndef SEMANTIC_RELOCALIZATION_ENGINE_H
#define SEMANTIC_RELOCALIZATION_ENGINE_H

// Tensorflow building scripts has been updated
// #include "tensorflow/core/platform/env.h"
// #include "tensorflow/core/public/session.h"

   #include <tensorflow/core/platform/env.h>
   #include <tensorflow/core/public/session.h>

// to support tensorflow serving
#include <tensorflow/cc/saved_model/loader.h>

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

#include <thread>
#include <mutex>
// not available in c++11, but defined in c++14
// #include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include "base/parallel_tasking_sched/threading_pool.hpp"
#include "base/parallel_tasking_sched/task.hpp"
#include "base/parallel_tasking_sched/channel.hpp"

using namespace svso::base::pts;

#ifndef OWNER_
#define OWNER_
struct Owner {

};
#endif

// py
namespace tf = tensorflow;

namespace svso {
namespace models {

/**
 * Checks if the given status is ok.
 * If not, the status is printed and the
 * program is terminated.
 */
static void checkStatus(const tensorflow::Status& status) {
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        exit(1);
    }
}

// general inference engine
class Engine {
public:
    using Ptr = std::shared_ptr<Engine>;
    using ConstPtr = std::shared_ptr<const Engine>;

    using InputTensors = std::vector<std::pair<tf::string, tf::Tensor>>;
    using OutputTensors = std::vector<tf::Tensor>;
    using ReturnType = tf::Status;
    using FutureType = std::future<ReturnType>;

    Engine(): graph_pb_def_(""), async_engine_executor_(nullptr), pool_(3) {}
    // Factors ...

    virtual ~Engine() {
        TearDown();
    }

    virtual void Init() = 0;

    virtual void TearDown() {}

    virtual FutureType Run(InputTensors& inputs, OutputTensors& outputs,
            const std::vector<string>& output_tensor_names,
            const std::vector<string>& target_node_names) = 0;

    virtual void Wait() {
        // async_engine_executor_->join();
    };

protected:
    virtual void HandleInference() = 0;

    std::string graph_pb_def_;
    std::thread* async_engine_executor_;
    ThreadPool pool_;

};

// Implementation for tensorflow inference engine using tensorflow c++ api (Tensorflow 2.2.0-rc). Implementation of inference engine by c_api was deprecated
class TensorFlowEngine: public Engine
{
public:
    using Ptr = std::shared_ptr<TensorFlowEngine>;
    using ConstPtr = std::shared_ptr<const TensorFlowEngine>;
    using Base = Engine;
    using Tensor = tf::Tensor;

    const std::string CLASS_NAME = "TensorFlowEngine";

    TensorFlowEngine() : Engine() {
        // async_engine_executor_ = new std::thread(&TensorFlowEngine::HandleInference, this);
    }
    // Factors ...

    // @todo : TODO
    explicit TensorFlowEngine(const std::string& graph_pb_def) : Engine() {
        load_graph(graph_pb_def);
        // async_engine_executor_ = new std::thread(&TensorFlowEngine::HandleInference, this);
        Init();
    }

    virtual ~TensorFlowEngine() {}

    void Init() override {
        const std::string& METHOD_NAME="Init";
        if (!graph_status_.ok()) {
            LOG(INFO) << base::logging::format(
                    "[%s.%s] TensorFlow graph is not defined! Use %s::load_graph(const std::string& graph_pb_def.)", CLASS_NAME.c_str(), METHOD_NAME.c_str()
                    );
        }

        // create session
        sess_status_ = tf::NewSession(tf::SessionOptions(), &session_);
        checkStatus(sess_status_);

        sess_status_ = session_->Create(graph_def_);
        checkStatus(sess_status_);
    }

    void TearDown() override {
        const std::string& METHOD_NAME="TearDown";
        if (session_) {
            LOG(INFO) << base::logging::format(
                    "[%s.%s] Close tensorflow session ...", CLASS_NAME.c_str(), METHOD_NAME.c_str()
                    );
            session_->Close();
            LOG(INFO) << base::logging::format(
                    "[%s.%s] Complete closing tensorflow session.", CLASS_NAME.c_str(), METHOD_NAME.c_str()
                    );

        }
    }

    void load_graph(const std::string& graph_pb_def)
    {
        const std::string& METHOD_NAME="load_graph";
        LOG(INFO) << base::logging::format(
                "[%s.%s] loading <%s> ...", CLASS_NAME.c_str(), METHOD_NAME.c_str(), graph_pb_def.c_str()
                );
        graph_status_ = ReadBinaryProto(tf::Env::Default(), graph_pb_def, &graph_def_);
        checkStatus(graph_status_);
        LOG(INFO) << base::logging::format(
                "[%s.%s] Complete loading <%s>.", CLASS_NAME.c_str(), METHOD_NAME.c_str(), graph_pb_def.c_str()
        );
        ///*
        int nd_count = graph_def_.node_size();
        for (int i = 0; i < nd_count; i++ )
        {
            auto node = graph_def_.node(i);
            std::cout << "Names: " << node.name() << std::endl;
        }
         //*/

    }

    //
    void load_saved_model(const std::string& export_dir)
    {
        const std::string& METHOD_NAME="load_graph";
        LOG(INFO) << base::logging::format(
                "[%s.%s] loading <%s> ...", CLASS_NAME.c_str(), METHOD_NAME.c_str(), export_dir.c_str()
        );
        // create session
        const char* tags = "serve";
        tf::SessionOptions options;
        tf::ConfigProto &config = options.config;
        // config tensorflow


        sess_status_ = tf::LoadSavedModel(tf::SessionOptions(options), tf::RunOptions(), export_dir, {tags}, &bundle_);
        checkStatus(sess_status_);

        session_ = bundle_.GetSession();

        // extract graph_def
        // see tf(v2.2.0) cc/saved_model/loader.cc, line 98
        graph_def_ = bundle_.meta_graph_def.graph_def();
        /*
        int nd_count = graph_def_.node_size();
        for (int i = 0; i < nd_count; i++ )
        {
            auto node = graph_def_.node(i);
            std::cout << "Names: " << node.name() << std::endl;
        }
        */
    }

    FutureType Run(InputTensors& inputs, OutputTensors& outputs,
            const std::vector<string>& output_tensor_names,
            const std::vector<string>& target_node_names) override {
        const std::string& METHOD_NAME="Run";
        if (async_engine_executor_) {
            LOG(INFO) << base::logging::format(
                    "[%s.%s] There is task to complete, waiting ...", CLASS_NAME.c_str(), METHOD_NAME.c_str()
            );
            async_engine_executor_->join();
            LOG(INFO) << base::logging::format(
                    "[%s.%s] Complete inference task.", CLASS_NAME.c_str(), METHOD_NAME.c_str()
            );
        }

        input_tensors_ = &inputs;
        output_tensors_ = &outputs;
        // Proceed to tensorflow inference task loop asynchronously, this is a cpu bounded routine, hence use thread directly
        FutureType fut = pool_.enqueue([=, this](){
            tf::Status status;
            if (output_tensors_ == nullptr) {
                LOG(ERROR) << base::logging::format(
                        "[%s.%s] <output_tensors> should not be nullptr!", CLASS_NAME.c_str(), METHOD_NAME.c_str()
                );
                return status;
            }
            LOG(INFO) << "enter into TF engine handler...";
            status = session_->Run(*input_tensors_, output_tensor_names, target_node_names, output_tensors_);
            LOG(INFO) << "complete running the graph...";
            checkStatus(status);
            LOG(INFO) << base::logging::format(
                    "[%s.%s] <output_tensor> number: %d", CLASS_NAME.c_str(), METHOD_NAME.c_str(), output_tensors_->size()
            );
            if (output_tensors_->size() == 0) {
                LOG(ERROR) << base::logging::format(
                        "[%s.%s] <output_tensors> is empty!", CLASS_NAME.c_str(), METHOD_NAME.c_str()
                );
            }
            return status;
        });
        return fut;
    }

    void Wait() override {
        Base::Wait();
    }

    InputTensors& input_tensors() {
        return *input_tensors_;
    }

    OutputTensors& output_tensors() {
        return *output_tensors_;
    }

protected:
    // @todo : TODO(move logics in TF_MRCNN_SemanticFeatureExtractor::detect here so to maintain the software coherence)
    virtual void HandleInference() override {

    }

    tf::Session* session_;
    tf::Status sess_status_;
    tf::GraphDef graph_def_;
    tf::SavedModelBundle bundle_;
    tf::Status graph_status_;
    // I/O Tensors
    InputTensors* input_tensors_;
    OutputTensors* output_tensors_;
};

    }
}


#endif //SEMANTIC_RELOCALIZATION_ENGINE_H
