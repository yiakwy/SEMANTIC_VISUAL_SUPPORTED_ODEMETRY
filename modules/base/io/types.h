//
// Created by yiak on 2021/7/1.
//

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_TYPES_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_TYPES_H

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
#include "env_config.h"

namespace svso {
namespace base {
namespace io {

using namespace svso::system;
using namespace svso::base::exceptions;
using namespace svso::base::logging;
namespace fs = boost::filesystem;

// This types will be used in ConfigManager, HDMapIO (visual part processing pipeline) and SVSO cpp POC project
using Scalar = boost::variant<int, size_t, float, double>;

class Number {
public:
    Scalar val;

    Number() {}
    template<typename type_t>
    explicit Number(type_t val_in): val(val_in) {};

    template<typename type_t>
    Number& operator=(type_t val_in) {
        val = val_in;
        return *this;
    }
};

class String {
public:
    std::string val;

    String() {}
    explicit String(const char* val_in) : val(val_in) {}
    explicit String(const std::string& val_in) : val(val_in) {}

    String& operator=(const char* val_in) {
        val = val_in;
        return *this;
    }

    String& operator=(const std::string& val_in) {
        val = val_in;
        return *this;
    }
};

class Boolean {
public:
    bool val;

    explicit Boolean() {}
    explicit Boolean(bool val_in) : val(val_in) {}

    Boolean& operator=(bool val_in) {
        val = val_in;
        return *this;
    }
};

// immutable
using Value = boost::variant<Number, String, Boolean>;

// mutable
class Class;
class Attributes;
class Array;
// @todo TODO Null is a thread-safe singleton
class Null {};


using Object = boost::variant<Value,
        boost::recursive_wrapper<Attributes>,
        boost::recursive_wrapper<Array>,
        Null>;
using ObjectPtr = std::shared_ptr<Object>;

class Attributes {
public:
    using TableType = std::unordered_map<std::string, ObjectPtr, std::hash<std::string>>;

    std::shared_ptr<Class> type;
    TableType attrs;

    Attributes() {}

    /*
     * @todo TODO attributes accessor
     * @todo TODO use concurrent hash map to enable setting and getting concurrently
     */
    void insert(const std::string& key, ObjectPtr val) {
        attrs[key] = val;
    }

    ObjectPtr get(const std::string& key) {
        TableType::const_iterator it = attrs.find(key);
        if (it != attrs.end()) {
            return it->second;
        } else {
            LOG(FATAL) << format("Cannot find key <%s> in ConfigManager attributes!", key.c_str());
        }
    }
};

class Array {
public:
    std::shared_ptr<Class> type;
    std::vector<ObjectPtr> values;

    Array() {}

    /*
     * @todo TODO array accessor
     */
};

class Class {

};

// lighter than Boost::Any
class Any {
public:
    Any() : holder_(nullptr) {}

    template <typename Class>
    Any(const Class& ins) : holder_(new InstanceHolder<Class>(ins)) {}

    Any(const Any& other) : holder_(other.holder_ ? other.holder_->Clone() : nullptr) {}

    ~Any() {
        if (holder_ != nullptr) {
            delete holder_;
            holder_ = nullptr;
        }

    }

    template<typename Class>
    Class* Cast() {
        return holder_ ? &(static_cast<InstanceHolder<Class>*>(holder_)->ins_) : nullptr;
    }

private:
    class PlaceHolder {
    public:
        virtual ~PlaceHolder() {}
        virtual PlaceHolder *Clone() const=0;
    };

    template<typename Class>
    class InstanceHolder : public PlaceHolder {
    public:
        using Type = InstanceHolder;

        explicit InstanceHolder(const Class& ins) : ins_(ins) {}
        virtual ~InstanceHolder() {}
        virtual PlaceHolder* Clone() const { return new Type(ins_);} // safely deleted by de-constructor of Any

        Class ins_;
    };

    PlaceHolder* holder_;
};

class AnyFactory {
public:

    using Type = AnyFactory;
    using Ptr = std::shared_ptr<AnyFactory>;

    AnyFactory() {}
    virtual ~AnyFactory() {}
    // implemented by sub_class any factory
    //   return Any( new Sub() );
    virtual Any NewInstance() { return Any(); }
};

    } // io
  } // base
} // svso
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_TYPES_H
