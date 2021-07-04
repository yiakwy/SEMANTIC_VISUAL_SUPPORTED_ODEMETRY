//
// Created by yiak on 2021/4/30.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_CONFIG_MANAGER_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_CONFIG_MANAGER_H

#ifdef __linux__
#include <regex.h>
#else
#error "Not defined in system different from Linux"
#endif

#include <memory>
// getenv
#include <cstdlib>

#include <algorithm>
#include <vector>
using std::vector;

#include <unordered_map>
using std::unordered_map;

#include <string>

#include <functional>
#include <type_traits>

#include "base/io/file.h"

// variant
#include <boost/variant.hpp>
#include <boost/filesystem.hpp>

#include "base/exceptions.h"
#include "base/logging.h"
#include "base/io/flags/config_manager_flags.h"
#include "base/io/types.h"
#include "env_config.h"

namespace svso {
namespace base {
namespace io {

/*
 * Usage example :
 *  base::ConfigManager::Ptr config_manager(new ConfigManager);
 *
 *  std::string root = config_manager->root();
 *
 *  config_manager->load(YOUR_CONF_PATH_FOR_THIS_MODULE);
 *
 *  std::string param_1;
 *  config_manager.get<std::string>("key_1", &param_1);
 *
 *  int param_2;
 *  config_manager.get<int>("key_2", &param_2);
 *
 *  size_t param_3;
 *  config_manager.get<int>("key_3", &param_3);
 *
 *  ...
 */
class ConfigManager {
public:
    using Type = ConfigManager;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    ConfigManager() {
        Init();
    }
    virtual ~ConfigManager() { Clear(); }

    void Clear() {}

    void Init() {
        root_ = env_config::CONF_DIR;
        root_dir_ = fs::path(root_);
        conf_root_ = FLAGS_conf_root;

        // compile all supported extension
        reader::compile_all_extensions();

    }

    std::string root() { return root_; }

    // @todo TODO restrict T to certain types : Number, String, Boolean ...
    void set_key(const std::string& key, Object val) {
        ObjectPtr objptr = std::make_shared<Object>(val);
        attrs_.insert(key, objptr);
    }

    void set_key(const std::string& key, ObjectPtr val) {
        attrs_.insert(key, val);
    }

    template<typename T>
    bool get(const std::string& key, T* ret, typename std::enable_if<std::is_same<T, int>::value, T>::type* = nullptr) {
        ObjectPtr obj = attrs_.get(key);
        if (std::is_arithmetic<T>::value) {
            Value value = boost::get<Value>(*obj);
            Number number = boost::get<Number>(value);
            *ret = boost::get<T>(number.val);
            return true;
        }
        return false;
    }

    template<typename T, typename std::enable_if<std::is_same<T, bool>::value, T>::type* = nullptr>
    bool get(const std::string& key, T* ret) {
        ObjectPtr obj = attrs_.get(key);

        if (std::is_same<T, bool>::value) {
            Value value = boost::get<Boolean>(*obj);
            Boolean boolean = boost::get<Boolean>(value);
            *ret = boolean.val;
            return true;
        }
        return false;
    }

    template<typename T>
    bool get(const std::string& key, T* ret, typename std::enable_if<std::is_same<T, std::string>::value, T>::type* = nullptr) {
        ObjectPtr obj = attrs_.get(key);

        if (std::is_same<T, std::string>::value) {
            Value value = boost::get<Value>(*obj);
            String str = boost::get<String>(value);
            *ret = str.val;
            return true;
        }
        return false;
    }

    template<typename T>
    bool get(const std::string& key, T* ret,
             typename std::enable_if<std::is_same<T,
                     std::vector<typename T::value_type,
                             typename T::allocator_type>
             >::value, T>::type* = nullptr) {
        ObjectPtr obj = attrs_.get(key);
        if (std::is_same<T, std::vector<typename T::value_type,
                typename T::allocator_type>>::value)
        {
            NOT_IMPLEMENTED
        }

        return false;
    }

    template<typename T>
    bool get(const std::string& key, T* ret,
             typename std::enable_if<std::is_same<T,
                     std::unordered_map<typename T::key_type,
                             typename T::mapped_type,
                             decltype(std::declval<T&>()[std::declval<const typename T::key_type&>()])>
             >::value, T>::type* = nullptr) {
        ObjectPtr obj = attrs_.get(key);
        if (std::is_same<T, std::unordered_map<typename T::key_type,
                typename T::mapped_type,
                decltype(std::declval<T&>()[std::declval<const typename T::key_type&>()])>>::value) {
            NOT_IMPLEMENTED
        }

        return false;
    }

    std::string get_absolute_path(const std::string& conf_path) {
        fs::path path(conf_path);
        if (path.is_absolute()) {
            return path.string();
        } else {
            return (root_dir_ / path).string();
        }
    }

    void load() {
        load(get_absolute_path(conf_root_));
    }

    void load(const std::string& conf_path, const std::string& file_type_anchor="yaml_path", const std::string skipped_path="conf", bool append_path=false)
    {
        fs::path path(conf_path);
        std::string extension = path.extension().string();

        if (is_yaml(extension)) {
            // is a yaml file
            load_yaml(conf_path, file_type_anchor, skipped_path, append_path);
        } else {
            NOT_IMPLEMENTED
        }
    }

    void load_module_conf_path() {
        std::string modules_def;
        get<std::string>("modules", &modules_def);

        // load modules
        load_yaml(modules_def, "prototxt", "", true);
    }

    bool is_yaml(const std::string& extension) {
        reader::FileExtension* ext = reader::parse_extension(extension.c_str());
        if (ext == nullptr) {
            return false;
        }

        if (ext->type == YAML_Symbol) {
            return true;
        }
        return false;
    }

    bool is_integer(const std::string& val) {
        reader::FileExtension* ext = reader::parse_extension(val.c_str());
        if (ext == nullptr) {
            return false;
        }

        if (ext->type == Integer_Symbol) {
            return true;
        }
        return false;
    }

    bool is_float_digits(const std::string& val) {
        reader::FileExtension* ext = reader::parse_extension(val.c_str());
        if (ext == nullptr) {
            return false;
        }

        if (ext->type == FLOAT_Symbol) {
            return true;
        }
        return false;
    }

protected:
    void load_yaml(const std::string& conf_path, const std::string& file_type_anchor="", const std::string& skipped_path="", bool append_path=false)
    {
        // will be replaced by "parseFromYaml" later
        YAML::Node root = YAML::LoadFile(conf_path);

        // add customer fields here
        if (!root[number_of_conf_key_].IsNull() && root[number_of_conf_key_].IsDefined()) {
            int numberOfFiles = root[number_of_conf_key_].as<int>();
            {
                Number val(numberOfFiles);
                set_key(number_of_conf_key_, val);
            }
        }

        std::vector<std::pair<std::string, YAML::Node>> check_list;
        YAML::Node conf = root;
        if (!root["conf"].IsNull() && root["conf"].IsDefined()) {
            conf = root["conf"];
        }

        std::string path_prefix = "";
        std::function<void(std::string, const YAML::Node&)> walk_yaml_tree = [=, &walk_yaml_tree,
                &check_list,
                &file_type_anchor,
                &skipped_path] (std::string path,
                                const YAML::Node& cur) -> void {
            if (cur.IsMap()) {
                for (YAML::const_iterator it = cur.begin(); it != cur.end(); ++it) {
                    std::string name = it->first.as<std::string>();
                    if (path != "" && path.compare(skipped_path) == 0) {
                        // skipped the path
                        path = "";
                    }
                    if (name.compare(file_type_anchor) != 0) {
                        fs::path prefix(path);
                        fs::path res(name);
                        walk_yaml_tree((prefix / res).string(), it->second);
                    } else {
                        walk_yaml_tree(path, it->second);
                    }

                }
            } else
            if (cur.IsSequence()) {
                for (YAML::const_iterator it = cur.begin(); it != cur.end(); ++it) {
                    walk_yaml_tree(path, *it);
                }
            } else
            if (cur.IsScalar()) {
                std::string name = path;
                check_list.emplace_back(name, cur);
            } else {
                LOG(INFO) << format("parsing eror, prefix : %s", path.c_str());
            }
        };

        // parsing general fields
        walk_yaml_tree(path_prefix, conf);

        // write config values back to the configuration
        for (auto kv : check_list) {
            std::string name = kv.first;
            YAML::Node scalar = kv.second;
            std::string buf = scalar.as<std::string>();

            // use regex to decide whether this value is string or number
            if (is_integer(buf)) {
                int number_raw = scalar.as<int>();

                Number val(number_raw);
                LOG(INFO) << format("Setting %s to path <%d>", name.c_str(), number_raw);

                set_key(name, val);
            } else
            if (is_float_digits(buf)) {
                float number_raw = scalar.as<double>();

                Number val(number_raw);
                LOG(INFO) << format("Setting %s to path <%f>", name.c_str(), number_raw);

                set_key(name, val);
            } else {
                // treat it as a path
                std::string yaml_path_raw = buf;

                if (append_path) {
                    fs::path prefix(name);
                    fs::path model_config_short_file(buf);
                    yaml_path_raw = (name / model_config_short_file).string();
                }

                std::string yaml_path = get_absolute_path(yaml_path_raw);

                String val(yaml_path);
                LOG(INFO) << format("Setting %s to path <%s>", name.c_str(), yaml_path.c_str());

                set_key(name, val);
            }

        }

    }

    void load_prototxt(const std::string& conf_path)
    {NOT_IMPLEMENTED}

    std::string number_of_conf_key_ = "numberOfConf";
    std::string root_;
    std::string conf_root_;
    fs::path root_dir_;
    Attributes attrs_;

    // supported suffix pattern, defaults to yaml, prototxt

    // load conf files for modules
};

    } // io
  } // base
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_CONFIG_MANAGER_H
