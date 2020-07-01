//
// Created by yiakwy on 4/8/20.
//

#ifndef SEMANTIC_RELOCALIZATION_SFE_H
#define SEMANTIC_RELOCALIZATION_SFE_H

#include <memory>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <unordered_map>
using std::pair;
using std::unordered_map;
#include <vector>
using std::vector;
#include <cmath>

#include "base/exceptions.h"
#include "engine.h"

using namespace svso::base::exceptions;
using namespace svso::models;
using tfe = TensorFlowEngine;

namespace svso {
    namespace models {

class Model {
public:
    using Ptr = std::shared_ptr<Model>;
    using ConstPtr = std::shared_ptr<const Model>;
    using ReturnType = TensorFlowEngine::ReturnType;

    Model() {}
    // Factors ...

    virtual ~Model() {
        TearDown();
    }

    virtual void TearDown() {}
protected:
    TensorFlowEngine::Ptr engine_;
};

class InferenceConfig {
public:

    int32_t IMAGE_MIN_DIM = 800;
    double IMAGE_MIN_SCALE = 0;
    int32_t IMAGE_MAX_DIM = 1024;
    std::string IMAGE_RESIZE_MODE = "square";
    std::vector<double> MEAN_PIXEL{123.7, 116.8, 103.9};
    int32_t NUM_CLASSES = 81;
};

// == Eigen Tensor Utility ==


// == Transplant Mask-RCNN utility codes to mold input images for neural network ==

Eigen::VectorXd compose_image_meta(const int32_t image_id, const Eigen::Vector3i& ori_shape, const Eigen::Vector3i& cur_shape,
                                   const Eigen::Vector4i& window, const double scale, const int32_t tracked_labels_set_size)
{
    Eigen::VectorXd meta = Eigen::VectorXd::Zero(1+3+3+4+1+tracked_labels_set_size);
    meta[0] = image_id;
    meta.segment(1, 3) = ori_shape.cast<double>();
    meta.segment(4, 3) = cur_shape.cast<double>();
    meta.segment(7, 4) = window.cast<double>();
    meta[11] = scale;
    return meta;
}

// see https://sourcegraph.com/github.com/matterport/Mask_RCNN/-/blob/mrcnn/utils.py#L388:1 for details
cv::Mat resize_image(const cv::Mat& img, const int32_t min_dim, const int32_t max_dim, const double min_scale, const std::string& mode,
        Eigen::Vector4i& window, double* scale, std::vector<std::pair<int, int>>& padding)
{
    int type = img.type();
    cv::Mat molded_img = img.clone();

    int H = img.rows;
    int W = img.cols;
    *scale = 1.0;
    padding.clear();
    padding.emplace_back(0, 0);
    padding.emplace_back(0, 0);
    padding.emplace_back(0, 0);
    window[2] = H;
    window[3] = W;

    if (mode == "" || mode == "none") {
        return molded_img;
    }

    if (min_scale) {
        *scale = fmax(1, min_dim / fmin(H, W) );
    }

    if (min_scale && *scale < min_scale)
    {
        *scale = min_scale;
    }

    // @todo (1) : TODO move string comparison to enum instance comparison
    if (max_dim && mode == "square")
    {
        int image_max = fmax(H, W);
        if (round(image_max * (*scale)) > max_dim) {
            *scale = (1.f * max_dim) / image_max;
        }
    }

    // Resize image using Spline interpolation, I prefer it (yiakwy)
    // For better understanding of preprocessing of images, see my implementation of Preprocessor for YoloV4 from notebook:
    // https://github.com/yiakwy/SpatialPerceptron/blob/master/notebooks/ModelArts-Improvement_One-Stage-Detectron.ipynb
    cv::resize(molded_img, molded_img, cv::Size(*scale * W, *scale * H), 0, 0, cv::INTER_CUBIC);

    // @todo (1)
    // padding and cropping the molded image
    if (mode == "square")
    {
        H = molded_img.rows;
        W = molded_img.cols;
        // padding
        int top_pad = (max_dim - H) / 2;
        int bottom_pad = max_dim- H - top_pad;
        int left_pad = (max_dim - W) / 2;
        int right_pad = max_dim - W - left_pad;
        padding[0] = std::make_pair(top_pad, bottom_pad);
        padding[1] = std::make_pair(left_pad, right_pad);
        cv::Scalar val(0,0,0);
        cv::copyMakeBorder(molded_img, molded_img, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, val);
        window << top_pad, left_pad, H + top_pad, W + left_pad;

    } else if (mode == "pad64") {
        NOT_IMPLEMENTED
    } else if (mode == "crop") {
        NOT_IMPLEMENTED
    } else {
        NOT_IMPLEMENTED
    }
    return molded_img;
}

// ==

std::string MatDtype(cv::Mat& img)
{
    int type = img.type();
    std::string type_ret;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch(depth) {
        case CV_8U:
            type_ret = "8U";
            break;
        case CV_8S:
            type_ret = "8S";
            break;
        case CV_16U:
            type_ret = "16U";
            break;
        case CV_16S:
            type_ret = "16S";
            break;
        case CV_32S:
            type_ret = "32S";
            break;
        case CV_32F:
            type_ret = "32F";
            break;
        case CV_64F:
            type_ret = "64F";
            break;
        default:
            type_ret = "User";
            break;
    }

    type_ret += "C";
    type_ret += (chans + '0');

    return type_ret;
}

class SemanticFeatureExtractor : public Model {
public:
    using Ptr = std::shared_ptr<SemanticFeatureExtractor>;
    using ConstPtr = std::shared_ptr<const SemanticFeatureExtractor>;

    SemanticFeatureExtractor() {

    }
    // Factors ...

    virtual ~SemanticFeatureExtractor() {
        TearDown();
    }

    void TearDown() override {

    }

    // @todo : TODO
    // see notebook/svso_tracker.ipynb
    // This is equivalent c++ version tranplant from MaskRCNN python implementation
    void mold_inputs(const std::vector<cv::Mat>& images, Eigen::Tensor<double, 4>& molded_images, Eigen::MatrixXd& image_metas, Eigen::MatrixXi& windows)
    {
        size_t l = images.size();
        for (int i=0; i < l; i++)
        {
            const cv::Mat& img = images.at(i);
            Eigen::Vector4i window;
            Eigen::VectorXd image_meta;
            double scale;
            std::vector<std::pair<int, int>> padding;
            cv::Mat molded_img = resize_image(img, config_.IMAGE_MIN_DIM, config_.IMAGE_MIN_SCALE, config_.IMAGE_MAX_DIM, config_.IMAGE_RESIZE_MODE,
                                               window, &scale,padding);
            molded_img.convertTo(molded_img, CV_64F);
            // substract mean value
            cv::subtract(molded_img, cv::Scalar(config_.MEAN_PIXEL[0], config_.MEAN_PIXEL[1], config_.MEAN_PIXEL[2]), molded_img);
            Eigen::Vector3i ori_shape{img.cols, img.rows, 3};
            Eigen::Vector3i cur_shape{molded_img.cols, molded_img.rows, 3};

            image_meta = compose_image_meta(0, ori_shape, cur_shape, window, scale, config_.NUM_CLASSES);

            // Append
            Eigen::Tensor<float, 4>::Dimensions dims = molded_images.dimensions();
            assert(dims[1] == cur_shape[0]);
            assert(dims[2] == cur_shape[1]);
            assert(dims[3] == cur_shape[2]);
            for (int j=0; j < dims[1]; j++) {
                for (int k=0; k < dims[2]; k++) {
                    cv::Vec3f intensity = molded_img.at<cv::Vec3f>(j,k);
                    molded_images(i,j,k,0) = intensity.val[0];
                    molded_images(i,j,k,1) = intensity.val[1];
                    molded_images(i,j,k,2) = intensity.val[2];
                }
            }

            windows.block(i, 0, 1, window.rows()) = window;
            image_metas.block(i,0,1, image_meta.rows()) = image_meta;

        }
    }

    // @todo : TODO
    // see notebook/svso_tracker.ipynb
    ReturnType detect(const cv::Mat& img, tfe::InputTensors* inputs, tfe::OutputTensors* outputs) {
        int H = img.rows;
        int W = img.cols;
        Eigen::Tensor<double, 4> molded_images(1, H, W, 3);
        Eigen::MatrixXi windows;
        Eigen::MatrixXd images_metas;

        std::vector<cv::Mat> images;
        images.push_back(img);

        windows.resize(1, 4);
        images_metas.resize(1, 1+3+3+4+1+config_.NUM_CLASSES);

        // mold input images
        mold_inputs(images, molded_images, images_metas, windows);

        // @todo : TODO
        // get anchors

        // @todo : TODO
        // create tensors


        // @todo : TODO
        // run base_engine_ detection
        // see examples from main.cpp, usage of TensorFlowEngine

    }

    // @todo : TODO
    void unmold_detections(tfe::OutputTensors* outputs)
    {

    }

    // @todo : TODO
    ReturnType compute()
    {

    }

    // @todo : TODO
    // see notebook/svso_tracker.ipynb
    ReturnType compute(const cv::Mat& img, const tfe::OutputTensors& detection, tfe::InputTensors* inputs, tfe::OutputTensors* outputs) {

    }

protected:
    // mrcnn model
    TensorFlowEngine::Ptr base_engine_;
    InferenceConfig config_;
};


    }
}

#endif //SEMANTIC_RELOCALIZATION_SFE_H
