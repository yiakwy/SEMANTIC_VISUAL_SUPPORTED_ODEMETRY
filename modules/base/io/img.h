//
// Created by yiak on 2020/8/27.
//
#pragma once

#ifndef SEMANTIC_RELOCALIZATION_IMG_H
#define SEMANTIC_RELOCALIZATION_IMG_H

#include <iostream>
#include <memory>
#include <string>

// @todo FIXME wrapper c libraies in a separate file

// GNU software
#include <glob.h>

// Linux libraries
#include <sys/stat.h>

#define _POXIS_SOURCE
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <unordered_map>
using std::unordered_map;
using std::pair;

#include <type_traits>

#include <boost/variant.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <glog/logging.h>
#include "base/logging.h"

#include "base/exceptions.h"

#include "env_config.h"

#include "devices/camera.pb.h"

using namespace svso::system;
using namespace svso::base::logging;
using namespace svso::base::exceptions;

namespace svso {
namespace base {
namespace io {
    namespace reader {

        // wrappers upon GNU "glob" and "globfree" funcs by overriden signatures
        static std::vector<std::string> glob(const std::string &ptn) {
            std::vector<std::string> files;
            glob_t glob_res;
            memset(&glob_res, 0, sizeof(glob_res));

            // retrieve files
            int status = glob(ptn.c_str(), GLOB_TILDE, nullptr, &glob_res);
            if (status != 0) {
                globfree(&glob_res);
                LOG(ERROR) << format("Failed to open %s", ptn.c_str());
                throw std::runtime_error(format("I/O failure: cannot open %s", ptn.c_str()));
            }

            for (size_t i = 0; i < glob_res.gl_pathc; ++i) {
                files.push_back(std::string(glob_res.gl_pathv[i]));
            }

            // clean up
            globfree(&glob_res);
            return files;
        }

        // wrapper to assert whether a file or directory exist
        bool is_exist(std::string &file_name) {
            return file_name != "" && boost::filesystem::exists(file_name);
        }

#ifndef FAILURE
#define FAILURE -1
#endif

        using RawImagePtr = std::shared_ptr<RawImage>;
        using RawImageConstPtr = std::shared_ptr<const RawImage>;

        RawImageConstPtr open_with_mmap(std::string fn) {
            size_t memory_size;
            RawImagePtr raw_image = nullptr;
            {
                if (!is_exist(fn)) {
                    LOG(ERROR) << format("Could not open the file %s", fn.c_str());
                    goto __error__;
                }

                int fd = open(fn.c_str(), O_RDONLY);
                if (fd == -1) {
                    LOG(ERROR) << format("Failure to open %s", fn.c_str());
                    close(fd);
                    goto __error__;
                }

                struct stat attrib;
                // read info from linux innode metadata
                if (stat(fn.c_str(), &attrib) < 0) {
                    LOG(ERROR)
                            << format("Could not find the meta data from the innode of the file[Linux] %s", fn.c_str());
                    close(fd);
                    goto __error__;
                }
                memory_size = (size_t) attrib.st_size;
                // map memory
                unsigned char *buf = static_cast<unsigned char *>(
                        ::mmap(0, memory_size, PROT_READ, MAP_SHARED, fd, 0)
                );

                raw_image.reset(
                        new RawImage()
                );
                std::string *data = raw_image->mutable_data();

                data->resize(memory_size);
                memcpy(&(*data)[0], &buf[0], memory_size);
                // unmap memory
                if (::munmap(buf, memory_size) == -1) {
                    LOG(ERROR) << "Failure to unmap memory";
                    close(fd);
                    goto __error__;
                } else {
                    close(fd);
                }
            }
            return (RawImageConstPtr)raw_image;

            __error__:
            // clean up
            exit(FAILURE);
        }

        /*
        * Internal Image representation in runtime
        */
        class Img {
        public:
            using Ptr = std::shared_ptr<Img>;
            using ConstPtr = std::shared_ptr<const Img>;

            Img() {}

            virtual ~Img() {}

            /*
            * attributes
            */


            Img(const Img &other) {}

            Img(Img &&other) noexcept
                    : Img() {
                *this = ::std::move(other);
            }

            inline Img &operator=(const Img &other) {
                CopyFrom(other);
                return *this;
            }

            inline Img &operator=(Img &&other) noexcept {
                if (this != &other) {
                    CopyFrom(other);
                }
                return *this;
            }

            void CopyFrom(const Img &other) {
                if (this == &other) return;
                Clear();

                data_ = other.data();

            }

            void Clear() {}

            /*
            * protected/private attributes accessor
            */
            const std::string &data() const { return data_; }

            const std::string &set_data(const std::string &data) {
                data_ = data;
                return data_;
            }

            cv::Mat mat(int type = CV_8UC3) {
                cv::Mat raw(1, static_cast<int>(data_.size()), type, (void *) data_.c_str());
                cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
                return img;
            }

            template<typename Scalar>
            cv::Mat mat() {
                cv::Mat raw(1, static_cast<int>(data_.size()), cv::traits::Type<Scalar>::value, (void *) data_.c_str());
                cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
                return img;
            }

            template<typename Scalar>
            typename Eigen::MatrixX<Scalar> matrix() {
                cv::Mat img = std::move(mat());
                Eigen::MatrixX<Scalar> mtx;
                if (!(mtx.Flags & Eigen::RowMajorBit)) {
                    NOT_IMPLEMENTED
                } else {
                    const cv::Mat tmp(img.rows, img.cols, cv::traits::Type<Scalar>::value,
                                      mtx.data(), (size_t) (mtx.outerStride() * sizeof(Scalar)));
                    img.convertTo(tmp, tmp.type());
                }
            }

        protected:
            std::string data_;

        };

    }

    // @todo : TODO
    namespace writer {}

}
}
}

#endif //SEMANTIC_RELOCALIZATION_IMG_H
