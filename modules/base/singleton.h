//
// Created by yiak on 2021/7/1.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_SINGLETON_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_SINGLETON_H

#include <typeinfo>
#include <memory>

namespace svso {
namespace base {

#ifdef __linux__
#include <pthread.h>
#else
#error "We havn't implemented for other system. PRs are welcome!"
#endif

// usually used together with ConfigManager :
//   base::ConfigManager::Ptr config_manager =
//      base::Singleton<base::ConfigManager>::get_instance();

// thread safe singleton.
template<class T>
class Singleton final {
public:

    using Type = Singleton;

    inline static T* instance_allocator(size_t size=1) {
        return new T();
    }
    inline static void instance_deallocator(T* instance) {
        if (instance != nullptr) {
            delete(instance);
        }
    }

    using T_ptr = std::shared_ptr<T>;

    static T_ptr get_instance() {
        pthread_once(&p_once_, &Type::new_instance);
        return instance_;
    }

private:
    // Not allowed to instantiate the instance of the class
    Singleton();
    ~Singleton();

    // @brief Construct the singleton instance
    static void new_instance() { instance_.reset(Singleton::instance_allocator(), &Singleton::instance_deallocator); }

private:
    static pthread_once_t p_once_;
    static T_ptr instance_;
};

  } // base
} // svso

#include "impl/singleton_impl.hpp"
#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_SINGLETON_H
