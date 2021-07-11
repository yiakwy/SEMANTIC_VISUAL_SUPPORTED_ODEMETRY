//
// Created by yiak on 2021/7/1.
//
#pragma once

#ifndef SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_KLASS_REGISTRY_H
#define SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_KLASS_REGISTRY_H

#include <memory>
#include <string>
#include <map>
using std::map;
using std::pair;

#include "base/io/types.h"
#include "base/logging.h"

// The module is adapted from Apollo.auto moudules/perception/lib/registerer.h
//  see: https://github.com/ApolloAuto/apollo/blob/master/modules/perception/lib/registerer/registerer.h
// Note the solution depends on GCC compiler "__attributes__" and we prefer a runtime implementation
// of a Class register here for simplicity.

namespace svso {
namespace base {

using namespace svso::base::logging;
using namespace svso::base::io;

/*
 * Usage example : Base* derived = X_Register.GetInstanceByName("DerivedClass");
 */
template<class BaseClass>
class Registry {
public:
    Registry() {}
    virtual ~Registry() {}

    using FactoryMap = std::map<std::string, AnyFactory::Ptr>;

    BaseClass* GetInstanceByName(const std::string& derived_klass) {
        auto it = sub_klasses.find(derived_klass);
        if (it == sub_klasses.end()) {
            LOG(FATAL) << format("Get instance %s failed", derived_klass.c_str());
        }
        // cast to BaseClass*
        Any any_ins = it->second->NewInstance();
        return *(any_ins.Cast<BaseClass*>());
    }

    // factory storage
    FactoryMap sub_klasses;

};

template<class BaseClass>
inline Registry<BaseClass>& get_registry() {
    static Registry<BaseClass> registry;
    return registry;
}

#define MODEL_REGISTER_SUBCLASS(base_klass, derived_klass)                     \
    namespace {                                                                \
      using namespace svso::base;                                           \
      class derived_klass##Factory : public AnyFactory {                       \
        public:                                                                \
          virtual ~derived_klass##Factory() {}                                 \
          virtual Any NewInstance() {                                          \
            return Any(new derived_klass());                                   \
          }                                                                    \
      };                                                                       \
      inline void Register##derived_klass##Facotry() {                         \
         Registry<base_klass>& registry = get_registry<base_klass>();          \
         Registry<base_klass>::FactoryMap& factory_map = registry.sub_klasses; \
         if (factory_map.find(#derived_klass) == factory_map.end()) {          \
           factory_map[#derived_klass].reset(new derived_klass##Factory());    \
         }                                                                     \
      }                                                                        \
      __attribute__((constructor)) void Register##derived_klass() {            \
         Register##derived_klass##Facotry();                                   \
      }                                                                        \
    } // anonymous namespace

  } // base
} // svso

#endif //SEMANTIC_VISUAL_SUPPORTED_ODEMETRY_KLASS_REGISTRY_H
