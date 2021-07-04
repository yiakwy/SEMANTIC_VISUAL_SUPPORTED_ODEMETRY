//
// Created by yiak on 2021/4/30.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_FILE_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_FILE_H

#ifdef __linux__
#include <regex.h>
#else
#error "Not defined in system different from Linux"
#endif

#include <string>
#include <functional>

#include <boost/variant.hpp>
#include <boost/filesystem.hpp>

// google protobuf parser
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
// YAML parser (see scripts/install_yaml.sh)
#include "yaml-cpp/yaml.h"

#include "base/io/blob_reader.h"

#include "base/exceptions.h"
#include "base/logging.h"

namespace svso {
namespace base {
namespace io {
namespace reader {

using namespace svso::base::logging;
using namespace svso::base::exceptions;

// Yaml Reader
/*
 * ProtoTypeConverter is a utility used to write YAML nodes of scalar type back to proto message body
 */
template<class ProtoType>
void parseFromYaml(const std::string& fn, ProtoType &pb) {
    // @todo TODO check fn is a yaml file

    // load and parse the configuration file
    YAML::Node root = YAML::LoadFile(fn);

    // @todo TODO convert <Key, Val> to protobuf message body
    try {
        if (root.IsNull()) {

        }

        pb = root.as<ProtoType>();

    } catch (YAML::InvalidNode & in) {
        // @todo TOOD
    } catch (YAML::Exception &e) {
        // @todo TODO
    }
}

// Protobuf Reader
/*
 * Supports both json, prototxt
 *
 */
template<class ProtoType>
bool parseFromProto(const std::string& fn, ProtoType &pb) {
    int fd = -1, ok = -1;
    Reader reader;

    size_t size = reader.get_length(fn, &fd, &ok);
    if (ok < 0) {
        LOG(FATAL) << format("Cannot open file %s", fn.c_str());
    }

    using namespace google::protobuf;
    using namespace google::protobuf::io;

    ZeroCopyInputStream *fin = new FileInputStream(fd);
    bool success = TextFormat::Parse(fin, &pb);
    if (!success) {
        LOG(FATAL) << format("Failed to read proto file %s!", fn.c_str());
    }
    delete fin;
    close(fd);
    return success;
}

// *** Relative path
std::string relative_to(const std::string& path, const std::string& target );

// *** File Extension

// FILE suffix
typedef struct _file_extension_t {
    const char* name; // YAML, Conf, Protobuf .... Currently we only support YAML and Protobuf prototxt format
    const char* def;
    int type; // 0: YAML, 1: PROTOBUF_TXT
    char* content;
    regex_t re;
} FileExtension;

// or equivalently to use enum class
#define YAML_Symbol 0
#define PROTOBUF_TXT_Symbol 1
// Note : Yaml-cpp cannot tell scalar between string and numbers
#define Integer_Symbol 2
#define FLOAT_Symbol 3

#define YAML_RE "^y(a)?ml$"
#define PROTOBUF_TXT_RE "^conf|prototxt$" // protobuf ASCII format
// adapted from　https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch06s10.html
#define INT_RE "^[-+]?[1-9][0-9]*$"
#define FLOAT_POINT_DIGITS_RE R"("^[-+]?[1-9][0-9]*\.[0-9]+[eE][-+]?[0-9]+$")"

static FileExtension file_extensions[] = {
        {"YAML", YAML_RE, YAML_Symbol},
        {"PROTOBUF", PROTOBUF_TXT_RE, PROTOBUF_TXT_Symbol},
        {"INTEGER", INT_RE, Integer_Symbol},
        {"FLOAT", FLOAT_POINT_DIGITS_RE, FLOAT_Symbol}
};

// compile them
#define EXEC_FAILURE 1
#define EXEC_SUCC 0
int compile_all_extensions();

FileExtension* parse_extension(const char* extension);

} // reader


namespace writer {

// Yaml Writer


// Protobuf Writer

} // writer

    } // base
  } // io
} // mapping
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_FILE_H
