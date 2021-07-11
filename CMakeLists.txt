cmake_minimum_required(VERSION 3.2)

project(SEMANTIC_VISUAL_SUPPORTED_ODEMETRY C CXX)

set (VERSION_MAJOR 1)
set (VERSION_MINOR 0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_FXX_FLAGS} -std=c++11")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# build type
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("[${PROJECT_NAME}] Build type: ${CMAKE_BUILD_TYPE}")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  # set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installed" CACHE PATH "default install apth" FORCE)
endif()

message("[${PROJECT_NAME}] Install path: ${CMAKE_INSTALL_PREFIX}")

# used by Apple xcode
if (EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
  execute_process( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -Wno-invalid-offsetof")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
# Add it for debug convenience, freshly added
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# setting compiling config

# by default build debug version
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif(NOT DEFINED CMAKE_BUILD_TYPE)

include(cmake/Utils.cmake)

if (UNIX OR APPLE)
  # suppose it is a darwin system
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
  # set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  set (LIB_LINKER_FLAGS "-dynamiclib")
  set (CMAKE_OS_NAME "MacOS" CACHE STRING "Operating sytem name" FORCE)
endif ()

# ignore the shape lib warning
if (CMAKE_HOST_APPLE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-tautological-compare")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-tautological-compare")
endif()
if (CMAKE_HOST_UNIX)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
endif()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

## Other OS BEGIN
echo ("Using libstdc++(gcc). You can change the tool chain with clang/clang++ in Ubuntu/CentOS/MacOS")

find_file(UBUNTU_EXTRA legal issue PATHS /etc)
if (UBUNTU_EXTRA)
  file (STRINGS ${UBUNTU_EXTRA} UBUNTU_FOUND REGEX Ubuntu)
  set (LIB_LINKER_FLAGS "-so")
  set (CMAKE_OS_NAME "Ubuntu" CACHE STRING "OS name" FORCE)
endif(UBUNTU_EXTRA)

if (EXISTS "/etc/redhat-release")
  set (LIB_LINKER_FLAGS "-so")
  set (CMAKE_OS_NAME "CentOS" CACHE STRING "OS name" FORCE)
endif()

## Other OS END

if (MSVC)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /utf-8")
  # enable parallel build for Visual Studio
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
endif(MSVC)

## General GCC / CLANG setting

# enable address asantinizer
if (GCC_SANTINIZER)
  if (CMAKE_HOST_UNIX)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -O1 -fno-omit-frame-pointer -g")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -O1 -fno-omit-frame-pointer -g")

  endif()
endif()

## add tests check target
add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND})
enable_testing()

## define generated files directory

if (DEFINED ENV{DATA_DIR})
 set (DATA_DIR "$ENV{DATA_DIR}")
else()
 # Mount the data directory to local "data" path
 set (DATA_DIR "${PROJECT_SOURCE_DIR}/data")
endif()

# set log_dir for glog
if (DEFINED ENV{LOG_DIR})
  set (LOG_DIR "$ENV{LOG_DIR}")
else ()
  set (LOG_DIR "${PROJECT_SOURCE_DIR}/log")
endif()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (SRCS_ROOT                      "${PROJECT_SOURCE_DIR}/modules")
set (ROOT                           "${PROJECT_SOURCE_DIR}")
set (CONF_DIR                       "${PROJECT_SOURCE_DIR}/config")
set (PROTO_GEN                      "${CMAKE_BINARY_DIR}/proto_codec")

list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

## Options
option(GLOBAL_DEBUG_SWITCH "Switch codes to debug mode" TRUE)
option(CUDA "Enable CUDA" ON)
option(GTEST "Build GTest test suites" ON)

## Import dependancies
include (cmake/Relies.cmake)
include (cmake/GRPC.cmake)
include (cmake/TensorFlow.cmake)

configure_file(config/env_config.h.in
  ${PROJECT_SOURCE_DIR}/config/env_config.h)
include_directories(config)
include_directories(modules)

# just for header files and libraries references
include_directories(vendors)

# add searching path for generated proto files
include_directories(${CODEC_GEN})

add_subdirectory(${SRCS_ROOT})
# add pysvso (python implementation of svso) and python bindings to svso core c++ libraries
add_subdirectory(cmake/pybind11)
add_subdirectory(${ROOT}/python)

# passed by cmake flags
if (GTEST)
  echo ("Building tests ...")
  enable_testing()
  add_subdirectory(tests/gtest)
else()
  echo ("GTEST flag is empty. Tests will not be built.")
  echo ("add -DGTEST=ON if you want build tests")
endif()
