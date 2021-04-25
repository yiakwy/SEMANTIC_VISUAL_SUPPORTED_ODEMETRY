//
// Created by yiak on 2021/4/17.
//

#ifndef SEMANTIC_RELOCALIZATION_BLOB_READER_H
#define SEMANTIC_RELOCALIZATION_BLOB_READER_H

#include <iostream>
#include <memory>
#include <string>

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

#include <glog/logging.h>
#include "base/logging.h"

namespace svso {
namespace base {
namespace io {
namespace reader {

using namespace svso::base::logging;
// apply node name convention
namespace fs = boost::filesystem;

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
static bool is_exist(const std::string &file_name) {
    return file_name != "" && fs::exists(file_name);
}

static bool is_directory(const std::string &file_name) {
    return is_exist(file_name) && fs::is_directory(file_name);
}

template<typename RawProto>
class BlobReader {
public:
    using RawProtoPtr = std::shared_ptr<RawProto>;
    using RawProtoConstPtr = std::shared_ptr<const RawProto>;

    BlobReader() {}
    virtual ~BlobReader() {
        Clear();
    }

    void Clear() {}

#ifndef FAILURE
#define FAILURE -1
#endif

    RawProtoConstPtr open_with_mmap(std::string fn) {
        size_t memory_size;
        RawProtoPtr raw_proto = nullptr;
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

            raw_proto.reset(
                    new RawProto()
            );
            std::string *data = raw_proto->mutable_data();

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
        return (RawProtoConstPtr)raw_proto;

        __error__:
        // clean up
        exit(FAILURE);
    }

};


} // reader
} // io
} // base
} // svso
#endif //SEMANTIC_RELOCALIZATION_BLOB_READER_H
