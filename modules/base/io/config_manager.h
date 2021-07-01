//
// Created by yiak on 2021/4/30.
//
#pragma once

#ifndef SEMANTIC_RELOCALIZATION_CONFIG_MANAGER_H
#define SEMANTIC_RELOCALIZATION_CONFIG_MANAGER_H

#include <memory>
#include <string>

namespace svso {
namespace base {
namespace io {

class ConfigManager {
public:

    using Type = ConfigManager;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    ConfigManager() {}
    virtual ~ConfigManager() {}



private:
    std::string root_;
};

}
} // base
} // svso
#endif //SEMANTIC_RELOCALIZATION_CONFIG_MANAGER_H
