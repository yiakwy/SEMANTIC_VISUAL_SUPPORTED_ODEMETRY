//
// Created by yiakwy on 4/8/20.
// Updated and Completed by yiakwy on 15/7/20
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

#include <functional>
#include <type_traits>

#include <future>
#include <thread>

// TensorFlow Inference Engine
#include "base/exceptions.h"
#include "engine.h"
#include "env_config.h"

// logging utilities
#include <glog/logging.h>
#include "base/logging.h"

// Eigen utility
#include "utils/eigen_util.h"

using namespace svso::base::exceptions;
using namespace svso::models;

using namespace svso::system;
using namespace svso::base::logging;

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
    std::vector<float> MEAN_PIXEL{123.7, 116.8, 103.9};
    int32_t NUM_CLASSES = 81;
    std::vector<int> RPN_ANCHOR_SCALES{31, 64, 128, 256, 512};
    std::vector<double> RPN_ANCHOR_RATIOS{0.5, 1., 2.};
    std::vector<int> BACKBONE_STRIDES{4, 8, 16, 32, 64};
    int RPN_ANCHOR_STRIDE = 1;
    std::string BACKBONE = "resnet101";
    // base model path, might be overridden by external flags
    std::string BASE_MODEL_DIR = "mrcnn_tmp"; // "mrcnn";
    std::string BASE_GRAPH_DEF = "mrcnn_tmp.pb";
    std::string SFE_MODEL_DIR = "sfe"; // "sfe_tmp";
    std::string SFE_GRAPH_DEF = "sfe.pb";
    std::string MODEL_DIR = format(
            "%s/models/coco", env_config::LOG_DIR.c_str()
    );
};

static std::string MatDtype(const cv::Mat& img)
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

// == Eigen Tensor Utility ==
/*
 * There are extensive tensor operations implemented in SemanticFeatureExtractor (SFE). The only offical reference
 * are :
 *   1. Eigen-unsupported (v3.3.9) : eigen.tuxfamily.org/dox-devel/unsupported/eigen_tensors.html
 *   2. Tensorflow cc api (v2.2.0) documentation
 */


// @todo TODO(avoid expensive copies) see the implementation of SemanticFeatureExtractor::detect
template <typename Type, size_t NUM_DIMS, int LAYOUT>
static void eigen_2_tensor(const Eigen::Tensor<Type, NUM_DIMS, LAYOUT>& eigen_val, tf::Tensor& t) {

}

// @todo TODO(avoid expensive copies)
template <typename Type, int NUM_DIMS, int LAYOUT>
static void eigen_2_tensor(const tf::Tensor& t, Eigen::Tensor<Type, NUM_DIMS, LAYOUT>& eigen_val)
{

}

// see stackoverflow :
template <typename Scalar, int NUM_DIMS, int LAYOUT>
static void eigen_tensor_2_matrix(const Eigen::Tensor<Scalar, NUM_DIMS, LAYOUT>& t, Eigen::MatrixX<Scalar>& matrix, int ROWS, int COLS)
{
    matrix = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, LAYOUT>> (t.data(), ROWS, COLS);
}

// == Transplant Mask-RCNN utility codes to mold input images for neural network ==

static Eigen::VectorXd compose_image_meta(const int32_t image_id, const Eigen::Vector3i& ori_shape, const Eigen::Vector3i& cur_shape,
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
static cv::Mat resize_image(const cv::Mat& img, const int32_t min_dim, const int32_t max_dim, const double min_scale, const std::string& mode,
        Eigen::Vector4i& window, double* scale, std::vector<std::pair<int, int>>& padding)
{
    int type = img.type();
    LOG(INFO) << "img type is: " << MatDtype(img);
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

    *scale = fmax(1, min_dim / fmin(H, W) );

    if (*scale < min_scale)
    {
        *scale = min_scale;
    }

    // @todo (1) : TODO move string comparison to enum instance comparison
    // [debug-1] scale == 1.67
    if (mode == "square")
    {
        int image_max = fmax(H, W);
        if (round(image_max * (*scale)) > max_dim) {
            *scale = (1.f * max_dim) / image_max;
        }
    }

    // Resize image using Spline interpolation (yiakwy)
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

class TF_MRCNN_SemanticFeatureExtractor : public Model {
public:
    using Ptr = std::shared_ptr<TF_MRCNN_SemanticFeatureExtractor>;
    using ConstPtr = std::shared_ptr<const TF_MRCNN_SemanticFeatureExtractor>;

    TF_MRCNN_SemanticFeatureExtractor() : config_(InferenceConfig()) {
        base_engine_.reset(new TensorFlowEngine() );
        const std::string base_model_dir = format(
                "%s/%s",
                config_.MODEL_DIR.c_str(),
                config_.BASE_MODEL_DIR.c_str()
        );
        base_model_dir_ = base_model_dir;
        // base_engine_->load_saved_model(base_model_dir);
        //*
        const std::string base_model_graph = format(
                "%s/%s",
                base_model_dir.c_str(),
                config_.BASE_GRAPH_DEF.c_str()
        );
        base_engine_->load_graph(base_model_graph);
        base_engine_->Init();
        //*/

        engine_.reset( new TensorFlowEngine() );
        const std::string sfe_model_dir = format(
                "%s/%s",
                config_.MODEL_DIR.c_str(),
                config_.SFE_MODEL_DIR.c_str()
        );
        sfe_model_dir_ = sfe_model_dir;
        engine_->load_saved_model(sfe_model_dir);

    }
    // Factors ...

    virtual ~TF_MRCNN_SemanticFeatureExtractor() {
        TearDown();
    }

    void TearDown() override {

    }

    // @todo : TODO
    // see notebook/svso_tracker.ipynb
    // This is equivalent c++ version transplant from Mask-RCNN python implementation
    void mold_inputs(const std::vector<cv::Mat>& images, Eigen::Tensor<float, 4, Eigen::RowMajor>& molded_images, Eigen::MatrixXd& image_metas, Eigen::MatrixXi& windows)
    {
        size_t l = images.size();
        bool first = true;
        for (int i=0; i < l; i++)
        {
            const cv::Mat& img = images.at(i);
            Eigen::Vector4i window;
            Eigen::VectorXd image_meta;
            double scale;
            std::vector<std::pair<int, int>> padding;
            cv::Mat molded_img = resize_image(img, config_.IMAGE_MIN_DIM, config_.IMAGE_MAX_DIM, config_.IMAGE_MIN_SCALE, config_.IMAGE_RESIZE_MODE,
                                               window, &scale,padding);
#ifndef NDEBUG
//          cv::imshow("molded_img", molded_img);
//          cv::waitKey(0);
            LOG(INFO) << "first pixel value(before conversion):" << std::endl << molded_img.at<cv::Vec3f>(0,0);
#endif
            // molded_img.convertTo(molded_img, CV_64F);
            molded_img.convertTo(molded_img, CV_32FC3);
#ifndef NDEBUG
            LOG(INFO) << "first pixel value(after):" << std::endl << molded_img.at<cv::Vec3f>(0,0);
#endif
            // substract mean value
            cv::subtract(molded_img, cv::Scalar(config_.MEAN_PIXEL[2], config_.MEAN_PIXEL[1], config_.MEAN_PIXEL[0]), molded_img);
#ifndef NDEBUG
            // show molded image see implementations(2019) from modules/io/image.h: ImageIOReader, readRawImageWithBuffer (decoded by cv::imdecode), modules/io/io.h: readBinaryMMap
            // Note for best performance we are not going to use cv::imread for several reasons:
            //  1. cv::imread does not utilise memory map technique for kernel to load data asynchronously and effectively
            //  2. we could get file information like size from operating system filesystem and leave other information to be parsed by decoder.
            //  3. we prefer lazy decoding and more often this is executed by another thread asynchronously
            // i.e. :
            //
            // cv::imshow("molded_img", molded_img);
            // cv::waitKey(0);
            LOG(INFO) << "first pixel value(completed):" << std::endl << molded_img.at<cv::Vec3f>(0,0);
#endif
            Eigen::Vector3i ori_shape{img.rows, img.cols, 3};
            Eigen::Vector3i cur_shape{molded_img.rows, molded_img.cols, 3};

            image_meta = compose_image_meta(0, ori_shape, cur_shape, window, scale, config_.NUM_CLASSES);

            if (first) {
                Eigen::array<int, 4> new_dims({int(l), cur_shape[0], cur_shape[1], cur_shape[2]});
                molded_images.resize(new_dims);
                first = false;
            }

            // Append
            Eigen::Tensor<int, 4>::Dimensions dims = molded_images.dimensions();

            for (int j=0; j < dims[1]; j++) {
                for (int k=0; k < dims[2]; k++) {
                    cv::Vec3d intensity = molded_img.at<cv::Vec3f>(j,k);
                    molded_images(i,j,k,0) = intensity.val[0];
                    molded_images(i,j,k,1) = intensity.val[1];
                    molded_images(i,j,k,2) = intensity.val[2];
                }
            }
#ifndef NDEBUG

            LOG(INFO) << "molded_images(0,0,0,:3): " << molded_images(0, 0, 0, 0) << " "
                                                     << molded_images(0, 0, 0, 1) << ' '
                                                     << molded_images(0, 0, 0, 2);
#endif
            windows.block(i, 0, 1, window.rows()) = window.transpose();
            image_metas.block(i,0,1, image_meta.rows()) = image_meta.transpose();

        }
    }

    // Note different from original paper using multi-scale anchors, since FPN take cares of scales by virtue of pyramid scales, the
    // Mask_RCNN implementation uses 1 scale at each of pyramid levels. See discussion #177 in Mask_RCNN repo for details.
    // Implementation reference:
    //   1. https://matterport/Mask_RCNN/utils/generate_pyramid_anchors
    //   2. d2l.ai/chapter_computer-vision/anchor.html
    Eigen::MatrixXd create_pyramid_anchors(const std::vector<int>& scales, const std::vector<double> ratios,
            const Eigen::MatrixXd& backbone_shapes,
            const std::vector<int>& backbone_strides,
            const int* anchor_stride,
            size_t* output_num) {

        // anchors shape : (N, (y1, x1, y2, x2)), N is equal to sum_{num_of_scales} { 1 * len(ratios) * out_nm_hi * out_num_wi }
        // Note in c++ Eigen
        Eigen::MatrixXd anchors;
        size_t H = 0, W = 0;
        size_t N = 0;

        for (int i=0; i < scales.size(); i++)
        {
            size_t Hi = backbone_shapes(i, 0), Wi = backbone_shapes(i, 1);
            H += Hi;
            W += Wi;
            N += ratios.size() * Hi * Wi;
        }

        // reserve memory for anchors
        // assert (N == 261888);
        anchors.resize(N, 4);

        size_t j=0;
        for (int i=0; i < scales.size(); i++)
        {
            size_t Hi = backbone_shapes(i, 0), Wi = backbone_shapes(i, 1);
            int scale = scales[i];
            int feature_stride = backbone_strides[i];
            for (int h=0; h < Hi; h++) {
                for (int w=0; w < Wi; w++) {
                    for (int k=0; k < ratios.size(); k++) {
                        double anchor_height = scale / std::sqrt(ratios[k]);
                        double anchor_width = scale * std::sqrt(ratios[k]);

                        // empty boxes filtered from computed result of the M-RCNN computation graph
                        double cy = h * (*anchor_stride) * feature_stride;
                        double cx = w * (*anchor_stride) * feature_stride;

                        double x1, y1, x2, y2;
                        y1 = cy - 0.5 * anchor_height;
                        x1 = cx - 0.5 * anchor_width;
                        y2 = cy + 0.5 * anchor_height;
                        x2 = cx + 0.5 * anchor_width;

                        Eigen::ArrayXd v(4);
                        v << y1, x1, y2, x2;
                        anchors.block<1, 4>(j++, 0) = v.transpose();
                    }
                }
            }
        }

        assert (N == j);
        *output_num = N;

        return anchors;

    }

    // detection results
    class DetectronResult {
    public:
        using Type = DetectronResult;

        Eigen::MatrixX<float> rois;
        std::vector<int> class_ids;
        std::vector<double> scores;
        std::vector<Eigen::MatrixX<int>> mask;

        Type filter(const std::vector<int>& excluded_ids) {
            Type ret;
            size_t size = rois.rows() - excluded_ids.size();
            ret.rois.resize(size, rois.cols());
            for (int i=0; i < size; i++)
            {
                size_t idx = i;
                if (std::binary_search(excluded_ids.begin(), excluded_ids.end(), idx)) {
                    continue;
                }

                ret.rois.row(i) = rois.row(idx);
                ret.class_ids.push_back(class_ids[idx]);
                ret.scores.push_back(class_ids[idx]);
                ret.mask.push_back(mask[idx]);
            }
            return ret;
        }
    };

    // @todo : TODO
    // see SemnaticFeatureDetector.detect from notebook/svso_tracker.ipynb
    tfe::FutureType detect(const cv::Mat& img, tfe::InputTensors* inputs, tfe::OutputTensors* outputs,
            std::vector<TF_MRCNN_SemanticFeatureExtractor::DetectronResult>& rets) {
        int H = img.rows;
        int W = img.cols;
        Eigen::Tensor<float, 4, Eigen::RowMajor> molded_images(1, H, W, 3);
        Eigen::MatrixXi windows;
        Eigen::MatrixXd images_metas;

        std::vector<cv::Mat> images;
        images.push_back(img);

        windows.resize(1, 4);
        images_metas.resize(1, 1+3+3+4+1+config_.NUM_CLASSES);

        // mold input images
        mold_inputs(images, molded_images, images_metas, windows);
#ifndef NDEBUG
      LOG(INFO) << "image metas:" << std::endl << eigen_utils::Eigen::pretty_print(images_metas);
#endif

        // @todo : TODO
        // get anchors
        Eigen::Vector3i image_shape;
        image_shape << H, W, 3;
        Eigen::Vector3i molded_shape;
        molded_shape << molded_images.dimension(1), molded_images.dimension(2), 3;
        Eigen::MatrixXd anchors = std::move( get_anchors(molded_shape) );
#ifndef NDEBUG
        // Eigen::MatrixXd tmp0 = anchors.block(0, 0, 10, 4);
      LOG(INFO) << "anchors(cpp computed):" << std::endl << eigen_utils::Eigen::pretty_print( anchors.block(0, 0, 10, 4) );
      LOG(INFO) << "python anchors snapshot(DEBUG):" << R"(
Out[1]:
array([[[-0.02211869, -0.01105934,  0.02114117,  0.01008183],
        [-0.01564027, -0.01564027,  0.01466276,  0.01466276],
        [-0.01105934, -0.02211869,  0.01008183,  0.02114117],
        ...,
        [ 0.5845174 ,  0.7614669 ,  1.2913378 ,  1.1143883 ],
        [ 0.68817204,  0.68817204,  1.1876833 ,  1.1876833 ],
        [ 0.7614669 ,  0.5845174 ,  1.1143883 ,  1.2913378 ]]],
      dtype=float32))";
#endif

        // @todo : TODO
        // create tensors
        // broadcast to (BATCH, anchors.rows(), anchor_shape.cols())
        // define tenors
        if (inputs == nullptr || outputs == nullptr) {
            LOG(FATAL) << "[Main] Wrong Value: inputs and outpus should not be null" << std::endl;
        }
        inputs->clear();
        outputs->clear();

        // In c++ it is also possible for tensorflow to create an reader operator to automatically read images from an image
        // path, where image tensor is built automatically and graph_def is finally converted from a variable of type tf::Scope.
        // In tensorflow, see codes defined in "tensorflow/core/framework/tensor_types.h" and "tensorflow/core/framework/tensor.h"
        // that users are able to use Eigen::TensorMap to extract values from the container for reading and assignment. (Lei (yiak.wy@gmail.com) 2020.7)
        tfe::Tensor _molded_images(tf::DT_FLOAT, tf::TensorShape({1, molded_shape(0), molded_shape(1), 3}));
        auto _molded_images_mapped = _molded_images.tensor<float, 4>();
        // @todo TODO using Eigen::TensorMap to optimize the copy operation, e.g.: float* data_mapped = _molded_images.flat<float>().data();  copy to the buf using memcpy
        //   ref: 1. discussion Tensorflow Github repo issue#8033
        //        2. opencv2 :
        //          2.1. grab buf: Type* buf = mat.ptr<Type>();
        //          2.2  memcpy cv::Mat to the buf
        //        3. Eigen::Tensor buffer :
        //          3.1 grab buf in RowMajor/ColMajor layout: tensor.data();
        //          3.2 convert using Eigen::TensorMap : Eigen::TensorMap<Eigen::Tensor<Type, NUM_DIMS>>(buf)
        //  _molded_images_mapped = Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>>(&data[0], 1, molded_shape_H, molded_shape_W, 3);
        for (int h=0; h < molded_shape(0); h++) {
            for (int w=0; w < molded_shape(1); w++) {
                _molded_images_mapped(0, h, w, 0) = molded_images(0, h, w, 2);
                _molded_images_mapped(0, h, w, 1) = molded_images(0, h, w, 1);
                _molded_images_mapped(0, h, w, 2) = molded_images(0, h, w, 0);
            }
        }
        LOG(INFO) << "_molded_images_mapped(0,0,0,:3): " << _molded_images_mapped(0, 0, 0, 0) << " "
                  << _molded_images_mapped(0, 0, 0, 1) << ' '
                  << _molded_images_mapped(0, 0, 0, 2);
        inputs->emplace_back("input_image", _molded_images);

        tfe::Tensor _images_metas(tf::DT_FLOAT, tf::TensorShape({1, images_metas.cols() } ) );
        auto _images_metas_mapped = _images_metas.tensor<float, 2>();
        for (int i=0; i < images_metas.cols(); i++)
        {
            _images_metas_mapped(0, i) = images_metas(0, i);
        }
        inputs->emplace_back("input_image_meta", _images_metas);

        tfe::Tensor _anchors(tf::DT_FLOAT, tf::TensorShape({1, anchors.rows(), anchors.cols()}));
        auto _anchors_mapped = _anchors.tensor<float, 3>();
        for (int i=0; i < anchors.rows(); i++)
        {
            for (int j=0; j < anchors.cols(); j++)
            {
                 _anchors_mapped(0,i,j) = anchors(i,j);
            }
        }
        inputs->emplace_back("input_anchors", _anchors);

        // @todo : TODO
        // run base_engine_ detection
        // see examples from main.cpp, usage of TensorFlowEngine

        // load saved_model.pb
//        tfe::FutureType fut = base_engine_->Run(*inputs, *outputs,
//                                                {"mrcnn_detection/Reshape_1:0", "mrcnn_class/Reshape_1:0", "mrcnn_bbox/Reshape:0", "mrcnn_mask/Reshape_1:0", "ROI/packed_2:0", "rpn_class/concat:0", "rpn_bbox/concat:0"}, {});
//        // load mrcnn.pb
//        tfe::FutureType fut = base_engine_->Run(*inputs, *outputs,
//                                                {"output_detections:0", "output_mrcnn_class:0", "output_mrcnn_bbox:0", "output_mrcnn_mask:0", "output_rois:0", "output_rpn_class:0", "output_rpn_bbox:0"}, {});
//        // load mrcnn_tmp.pb
        tfe::FutureType fut = base_engine_->Run(*inputs, *outputs,
                                                {"mrcnn_detection/Reshape_1:0", "mrcnn_class/Reshape_1:0", "mrcnn_bbox/Reshape:0", "mrcnn_mask/Reshape_1:0", "ROI/packed_2:0", "rpn_class/concat:0", "rpn_bbox/concat:0"}, {});

        // pass fut object to anther thread by value to avoid undefined behaviors
        std::shared_future<tfe::ReturnType>  fut_ref( std::move(fut) );

        // wrap fut with a new future object and pass local variables in
        std::future<ReturnType> wrapped_fut = std::async(std::launch::async, [=, &rets]() -> ReturnType {
            LOG(INFO) << "enter into sfe TF handler ...";

            // fetch result
            fut_ref.wait();

            tf::Status status = fut_ref.get();
            std::string graph_def = base_model_dir_;
            if (status.ok()) {

                if (outputs->size() == 0) {
                    LOG(INFO) << format("[Main] Found no output: %s!", graph_def.c_str(), status.ToString().c_str());
                    return status;
                }
                LOG(INFO) << format("[Main] Success: infer through <%s>!", graph_def.c_str());
                // @todo : TODO fill out the detectron result

                tfe::Tensor detections = (*outputs)[0];
                tfe::Tensor mrcnn_mask = (*outputs)[3];

                // @todo : TODO convert tf::Tensor to eigen matrix/tensor
                auto detections_mapped = detections.tensor<float, 3>();
                auto mrcnn_mask_mapped = mrcnn_mask.tensor<float, 5>();
#ifndef NDEBUG
                LOG(INFO) << format("detections[0,0,:](shape:(%d,%d,%d)):",
                        detections_mapped.dimension(0),
                        detections_mapped.dimension(1),
                        detections_mapped.dimension(2))
                << std::endl << detections_mapped.chip(0, 0).chip(0, 0);
                // LOG(INFO) << "mask:" << std::endl << mrcnn_mask_mapped;

#endif
                for (int i=0; i < images.size(); i++) {
                    // Eigen::Tensor is default ColMajor layout, which is different from c/c++ matrix layout.
                    // Note only column layout is fully supported for the moment (v3.3.9)
//                    Eigen::Tensor<float, 2> detection = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 2, Eigen::RowMajor>>
//                    (detections_mapped.chip(i, 0));
                    Eigen::Tensor<float, 2, Eigen::RowMajor> detection = detections_mapped.chip(i, 0);
                    // Generate mask using a threshold
//                    Eigen::Tensor<float, 4> mask = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 4, Eigen::RowMajor>>
//                    (mrcnn_mask_mapped.chip(i, 0));
                    Eigen::Tensor<float, 4, Eigen::RowMajor> mask = mrcnn_mask_mapped.chip(i, 0);


                    DetectronResult ret;
                    Eigen::MatrixXi window = windows.row(i);

                    unmold_detections(detection, mask, image_shape, molded_shape, window, ret);
                    rets.push_back( std::move(ret) );
                }

            } else {
                LOG(INFO) << format("[Main] Failed to infer through <%s>: %s!", graph_def.c_str(), status.ToString().c_str());
            }
            return status;
        });

        return wrapped_fut;

    }

    // @todo : TODO(ColMjaor)
    void unmold_detections(const Eigen::Tensor<float, 2, Eigen::RowMajor>& detection, const Eigen::Tensor<float, 4, Eigen::RowMajor>& mask,
            const Eigen::Vector3i& image_shape, const Eigen::Vector3i& molded_shape,
            const Eigen::MatrixXi& window,
            DetectronResult& ret)
    {
        // @todo : TODO (avoid expensive copies)
        // extract non-zero boxes
        size_t N;
        // find first bbox which is zero
#ifndef NDEBUG
        Eigen::array<int, 2> offset0 = {0, 0};
        Eigen::array<int, 2> size0 = {100, 6};
        Eigen::Tensor<float, 2, Eigen::RowMajor> sub_view = std::move( detection.slice(offset0, size0) );
        LOG(INFO) << "subview of detection:" << std::endl << sub_view;
#endif
        {
            int i;
            for (i = 0; i < detection.dimension(0); i++) {
                Eigen::Tensor<float, 1, Eigen::RowMajor> tmp = detection.chip(i, 0).abs();
                if (abs(tmp(4)) < 1e-6) {
                    N = i;
                    break;
                }
            }
            if (i >= detection.dimension(0)) {
                N = detection.dimension(0);
            }
        }
        ret.rois.resize(N, 4);
        Eigen::array<int, 2> offset1 = {0, 0};
        Eigen::array<int, 2> size1 = {int(N), 4};
        Eigen::Tensor<float, 2, Eigen::RowMajor> sub_detection = std::move( detection.slice(offset1, size1) );
        eigen_tensor_2_matrix<float, 2>(sub_detection, ret.rois, int(N), 4);
#ifndef NDEBUG
        LOG(INFO) << "ret.rois" << std::endl << eigen_utils::Eigen::pretty_print(ret.rois);
#endif

        std::vector<Eigen::MatrixX<float>> numeric_matrice;
        for (int i=0; i < N; i++) {
            int class_id = int(detection(i, 4));
            ret.class_ids.push_back( class_id );
            ret.scores.push_back( detection(i, 5) );

            // extract mask
            Eigen::MatrixX<float> numeric_mat;

            Eigen::Tensor<float, 3, Eigen::RowMajor> mask_slices = std::move( mask.chip(i, 0) );
            Eigen::Tensor<float, 2, Eigen::RowMajor> _mask = std::move( mask_slices.chip(class_id, 2) );
            eigen_tensor_2_matrix<float, 2>(_mask, numeric_mat, mask.dimension(0), mask.dimension(1));

            numeric_matrice.push_back( std::move(numeric_mat) );
        }
        // @todo : TODO
        // "converted to boolean values"
        for (int i=0; i < N; i++)
        {
            float* mask_buf = numeric_matrice[i].data();
            cv::Mat raw_mask_img(cv::Size(mask.dimension(1), mask.dimension(0)), CV_32FC1, mask_buf);
            cv::Mat mask_img(raw_mask_img.size(), CV_8UC1);
            cv::threshold(raw_mask_img, mask_img, 0.5, 255, cv::THRESH_BINARY);
            Eigen::MatrixX<int> mask_numeric;
            cv::cv2eigen(mask_img, mask_numeric);

#ifndef NDEBUG
            // Eigen::MatrixXi tmp1 = mask_numeric.block(0,0,1,5);
            LOG(INFO) << format("subview of numeric_matrice[0]: %s", eigen_utils::Eigen::pretty_print( numeric_matrice[0].block(0, 0, 1, 5) ).c_str() );
            LOG(INFO) << format("subview of mask_numeric: %s", eigen_utils::Eigen::pretty_print( mask_numeric.block(0,0,1,5) ).c_str() );
#endif

            ret.mask.push_back(mask_numeric);
        }

        // @todo : TODO (hard)
        // "Translate normalized coordinates in the resized image to pixel
        // coordinates in the original image before resizing"
        auto norm_boxes = [=] (const Eigen::MatrixXi& box, const Eigen::Vector3i& shape) -> Eigen::MatrixXf {
            int H = shape(0), W = shape(1);
            Eigen::Vector4f factor{H-1, W-1, H-1, W-1};
            Eigen::Vector4i offset{0, 0, 1, 1};
            // normalized coordiantes
            Eigen::MatrixXf ret = ( box.cast<float>().rowwise() - offset.cast<float>().transpose() ).array().rowwise() / factor.transpose().array();
            // LOG(INFO) << eigen_utils::Eigen::pretty_print(ret);
            return ret;
        };
        auto denorm_boxes = [=] (const Eigen::MatrixXf& box, const Eigen::Vector3i& shape) -> Eigen::MatrixXf {
            int H = shape(0), W = shape(1);
            Eigen::Vector4f factor{H-1, W-1, H-1, W-1};
            Eigen::Vector4f offset{0, 0, 1, 1};
            Eigen::MatrixXf ret = ( box.array().rowwise() * factor.transpose().array() ) ;
            ret = ret.rowwise() + offset.transpose();
            return ret;
        };
#ifndef NDEBUG
        LOG(INFO) << "window: " << eigen_utils::Eigen::pretty_print(window);
#endif
        Eigen::MatrixXf new_window = norm_boxes(window, molded_shape);
#ifndef NDEBUG
        LOG(INFO) << "new_window: " << eigen_utils::Eigen::pretty_print(new_window);
        LOG(INFO) << "python new window: " << R"(
Out[2]: array([0.12512219, 0.        , 0.8748778 , 1.        ], dtype=float32)
)";
#endif
        // "Convert boxes to normalized coordinates on the window"
        float win_h = new_window(2) - new_window(0);
        float win_w = new_window(3) - new_window(1);
        Eigen::Vector4f factor{win_h, win_w, win_h, win_w};
        Eigen::Vector4f offset{new_window(0, 0), new_window(0, 1), new_window(0, 0), new_window(0, 1)};
#ifndef NDEBUG
        LOG(INFO) << "ret.rois(0,1): " << ret.rois(0,1);
        LOG(INFO) << "offset(1): " << offset(1);
        LOG(INFO) << "factor(1): " << factor(1);
        LOG(INFO) << "tmp : " << ( ret.rois(0,1) - offset(1) ) / factor(1);
#endif
        ret.rois = ( ret.rois.rowwise() - offset.transpose() ).array().rowwise() / factor.transpose().array();

#ifndef NDEBUG
        LOG(INFO) << "rois bbox in normalized coordinates on the window : " << std::endl << eigen_utils::Eigen::pretty_print(ret.rois);
#endif

        // "Convert boxes to pixel coordinates on the original image(with image_shape)"
        ret.rois = denorm_boxes(ret.rois, image_shape);

#ifndef NDEBUG
        Eigen::MatrixXi tmp0 = std::move( ret.rois.cast<int>() );
        LOG(INFO) << "denormed mask: " << std::endl << eigen_utils::Eigen::pretty_print( tmp0 );
        LOG(INFO) << "python mask snapshot(DEBUG):" << R"(
Out[1]:
array([[  0, 227, 123, 358],
       [  3,   3, 155, 160],
       [292, 292, 348, 338],
       [318, 541, 364, 598],
       [327, 332, 469, 591],
       [177,  15, 256, 150],
       [255,  21, 343, 132],
       [157,  22, 209, 108],
       [ 69, 295, 304, 602]], dtype=int32)


)";
#endif

        // @todo : TODO
        // "Filter out detections with zero area. Happens in early training when
        // network weights are still random"
        std::vector<int> exclude_ids;
        for (int i=0; i < N; i++)
        {
            Eigen::MatrixXi box = ret.rois.row(i).cast<int>();
            double area = (box(0,2) - box(0,0)) * (box(0,3) - box(0,1));
            if (area < 0) {
                exclude_ids.push_back(i);
            }
        }
        DetectronResult filtered_ret = ret.filter(exclude_ids);

        // @todo : TODO
        // "Resize masks to original image size"
        for (int i=0; i < filtered_ret.rois.rows(); i++)
        {
            Eigen::MatrixXi mask_numeric = filtered_ret.mask[i];
            Eigen::MatrixXi bbox = filtered_ret.rois.row(i).cast<int>();
            cv::Mat mask_img(cv::Size(mask_numeric.cols(), mask_numeric.rows()), CV_8UC1, mask_numeric.data());
            double H, W;
            H = bbox(0,2) - bbox(0, 0);
            W = bbox(0, 3) - bbox(0, 1);
            cv::resize(mask_img, mask_img, cv::Size(W, H), 0, 0, cv::INTER_CUBIC);
            cv::Mat full_mask(cv::Size(image_shape[1], image_shape[0]), CV_8UC1, cv::Scalar(0));
            mask_img.copyTo(full_mask(cv::Rect(bbox(0,1), bbox(0,0), W, H)));
            cv2eigen(full_mask, filtered_ret.mask[i]);
        }

        ret = filtered_ret;
    }

    Eigen::MatrixXd get_backbone_shapes(int H, int W) {
        assert (config_.BACKBONE == "resnet101");
        int N = config_.BACKBONE_STRIDES.size();
        int M = 2;
        Eigen::MatrixXd backbone_shapes;
        backbone_shapes.resize(N, M);

        for (int i=0; i < N; i++) {
            backbone_shapes(i,0) = int( std::ceil(H / config_.BACKBONE_STRIDES[i] ) );
            backbone_shapes(i,1) = int( std::ceil(W / config_.BACKBONE_STRIDES[i] ) );
        }

        return backbone_shapes;
    }

    // @todo : TODO
    Eigen::MatrixXd get_anchors(Eigen::Vector<int, 3> image_shape) {
        // get computed backbone shapes
        Eigen::MatrixXd backbone_shapes = std::move( get_backbone_shapes(image_shape(0), image_shape(1)) );

#ifndef NDEBUG
        LOG(INFO) << format("backbone_shapes: %s", eigen_utils::Eigen::pretty_print(backbone_shapes).c_str());
#endif

        // generate anchors
        if (this->anchors_cache_.find(image_shape) == this->anchors_cache_.end()) {
            size_t output_num;
            Eigen::MatrixXd anchors = std::move( create_pyramid_anchors(
                        config_.RPN_ANCHOR_SCALES,
                        config_.RPN_ANCHOR_RATIOS,
                        backbone_shapes,
                        config_.BACKBONE_STRIDES,
                        &config_.RPN_ANCHOR_STRIDE,
                        &output_num
                    ) );

            // norm boxes
            std::vector<int> factor{image_shape(0)-1, image_shape(1)-1};
            for (int i=0; i < output_num; i++)
            {
                double x1, y1, x2, y2;
                y1 = anchors(i, 0);
                x1 = anchors(i, 1);
                y2 = anchors(i, 2);
                x2 = anchors(i, 3);

                anchors(i, 0) = y1 / factor[0];
                anchors(i, 1) = x1 / factor[1];
                anchors(i, 2) = (y2 - 1) / factor[0];
                anchors(i, 3) = (x2 - 1) / factor[1];
            }

            anchors_cache_[image_shape] = anchors;

        }

        return anchors_cache_.find(image_shape)->second;

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

    std::string base_model_dir_;
    std::string sfe_model_dir_;

    using ANCHOR_CACHE_KEY = Eigen::Vector<int, 3>;
    using ANCHOR_CACHE_VAL = Eigen::MatrixXd;

    // see boost.hash_combine for details of algorithms for hash projector
    template<typename T>
    struct _hasher : std::unary_function<T, size_t> {
        std::size_t operator()(T const & shape) const {
            size_t ret = 0;
//            if (std::is_same<T, Eigen::Tensor<int, 1>>::value) {
//                typename T::Dimensions dims = shape.dimensions();
//                for (size_t i=0; i < dims[0]; i++) {
//                    ret ^= std::hash<typename T::Scalar>()( shape(i) ) + 0x9e3779b9 + (ret << 6) + (ret >> 2);
//                }
//            } else {
//                // treat T as Eigen::Matrix<Scalar, ...>
//                for (size_t i=0; i < shape.size(); i++)
//                {
//                    ret ^= std::hash<typename T::Scalar>()( shape(i) ) + 0x9e3779b9 + (ret << 6) + (ret >> 2);
//                }
//            }
            for (size_t i=0; i < shape.size(); i++)
            {
                ret ^= std::hash<typename T::Scalar>()( shape(i) ) + 0x9e3779b9 + (ret << 6) + (ret >> 2);
            }
            return ret;
        }
    };

    std::unordered_map<ANCHOR_CACHE_KEY, ANCHOR_CACHE_VAL, _hasher<ANCHOR_CACHE_KEY > > anchors_cache_;

};


    }
}

#endif //SEMANTIC_RELOCALIZATION_SFE_H
