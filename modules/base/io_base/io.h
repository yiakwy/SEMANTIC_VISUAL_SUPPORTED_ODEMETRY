//
// Created by yiak on 2020/8/27.
//
#pragma once

#ifndef SEMANTIC_RELOCALIZATION_IO_H
#define SEMANTIC_RELOCALIZATION_IO_H

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

#include <boost/variant.hpp>
#include <boost/filesystem.hpp>

#include <glog/logging.h>
#include "base/logging.h"

#include "env_config.h"

using namespace svso::system;
using namespace svso::base::logging;

namespace svso {
namespace base {
namespace io_base {

    // wrappers upon GNU "glob" and "globfree" funcs by overriden signatures
    static std::vector<std::string> glob(const std::string& ptn)
    {
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

        for (size_t i=0; i < glob_res.gl_pathc; ++i) {
            files.push_back(std::string(glob_res.gl_pathv[i]));
        }

        // clean up
        globfree(&glob_res);
        return files;
    }

    // wrapper to assert whether a file or directory exist
    bool is_exist(std::string& file_name) {
        return boost::filesystem::exists(file_name);
    }

#ifndef FAILURE
#define FAILURE -1
#endif

    class RawImage {

    };



}
}
}

#endif //SEMANTIC_RELOCALIZATION_IO_H
