# add tests
include(CTest)
include("cmake/External/GTest.cmake")

# add logs
include("cmake/External/GLog.cmake")

# add flags
include("cmake/External/GFlags.cmake")

# Mapping externals used for this project
echo ("Home directory : $ENV{HOME}")
if (DEFINED ENV{MAPPING_EXTERNAL_DIR})
    set (EXTERNAL_DIR "$ENV{MAPPING_EXTERNAL_DIR}")
else ()
    set (EXTERNAL_DIR "$ENV{HOME}/mapping_external")
    message (WARNING "System variable EXTERNAL_DIR is nonexist, using ${EXTERNAL_DIR} instead.")
endif ()

# used to install compiled third party libraries by other groups
# in this project, this should be empty
if (NOT IS_DIRECTORY ${EXTERNAL_DIR})
    message(FATAL_ERROR "EXTERNAL_DIR ${EXTERNAL_DIR} is not a directory")
endif ()

if (CMAKE_HOST_WIN32)
    set(EXTERNAL_LIBS_DIR ${EXTERNAL_DIR}/win64)
elseif (CMAKE_HOST_APPLE)
    set(EXTERNAL_LIBS_DIR ${EXTERNAL_DIR}/darwin)
elseif(CMAKE_HOST_UNIX)
    if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(EXTERNAL_LIBS_DIR ${EXTERNAL_DIR}/linux)
    elseif (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(EXTERNAL_LIBS_DIR ${EXTERNAL_DIR}/arm64-linux)
    endif()
endif()
echo("EXTERNAL_LIBS_DIR : ${EXTERNAL_LIBS_DIR}")

# OpenCV
# To avoid conflicts introduced by opencv4, I installed the package frm source
# see vendors/github.com/opencv and installer scripts/thirdparties/linux/deb/apt/install_opencv.sh
set (OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV 4.0 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV 3.0 QUIET)
  if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV >= 3.0 not found")
  endif()
endif()
if (OpenCV_FOUND)
  echo("Find Opencv (ver.${OPENCV_VERSION}) (include: ${OpenCV_INCLUDE_DIRS}, library: ${OpenCV_LIBRARIES})")
  include_directories(
    ${OpenCV_INCLUDE_DIRS}
    )
endif()

# Boost
find_package(Boost COMPONENTS system filesystem REQUIRED)

# Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# nlohmann
# include_directories(${EXTERNAL_DIR}/include/nlohmann)

# Geos
# set(GEOS_INCLUDE_DIRS ${EXTERNAL_LIBS_DIR}/geos/include)
# set(GEOS_LIBS_DIR ${EXTERNAL_LIBS_DIR}/geos/lib)
# set(GEOS_LIBRARIES geos)
# include_directories(${GEOS_INCLUDE_DIRS})
# link_directories(${GEOS_LIBS_DIR})

# GeographicLib
# set(GeographicLib_LIBRARIES Geographic)
# include_directories(${GeographicLib_INCLUDE_DIRS})
# link_directories(${GeographicLib_LIBRARY_DIRS})

# Eigen
# Tensorflow 2.2.rc uses modified version Eigen 3.3.9 maintained by bazel compilation system
# set (Eigen_INSTALL "/usr/local/include/eigen3")
# set (Eigen_INSTALL "/home/yiakwy/.cache/bazel/_bazel_yiakwy/729cb3000927cd0e322e439998a82145/external/eigen_archive")
find_package(Eigen3 3.3.9 REQUIRED)
# Eigen delivers Eigen3Config.cmake since V3.3.3
# Modern way CMake dependencies
# find_package(Eigen3 3.3 CONFIG REQUIRED)
if (Eigen3_FOUND)
  echo ("Found Eigen3 (ver.${EIGEN3_VERSION}) (include: ${EIGEN3_INCLUDE_DIR})")
else()
  echo ("Eigen3 not found!")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# ceres
# set(Ceres_DIR ${EXTERNAL_LIBS_DIR}/ceres/lib/cmake/Ceres)
# find_package(Ceres QUIET REQUIRED)
# include_directories(${CERES_INCLUDE_DIRS})

# Since we switch to gcc > 8.2, c++14, we decided to move to PCL-1.11 to replace old version of PCL-1.8
# Though we have great efforts in optimizing codes base pertaining to PCL, PCL-1.11 still have a better native support to
# CUDA and threaded algorithms in fundamental components.
# PCL-1.11
set(PCL_DIR ${EXTERNAL_LIBS_DIR}/pcl/share/pcl-1.11)
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    find_package(PCL REQUIRED
            COMPONENTS common kdtree octree search surface io ml sample_consensus filters geometry 2d features segmentation visualization registration)
else ()
    find_package(PCL QUIET REQUIRED
            COMPONENTS common kdtree octree search surface io ml sample_consensus filters geometry 2d features segmentation registration)
endif ()
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
list(APPEND CMAKE_INSTALL_RPATH "${PCL_LIBRARY_DIRS}")

#[[
# GTSAM
set(GTSAM_DIR ${EXTERNAL_LIBS_DIR}/gtsam)
find_package(GTSAM REQUIRED) # Uses installed package
include_directories(${GTSAM_INCLUDE_DIR})
]]

# TBB
LIST(APPEND CMAKE_MODULE_PATH ${EXTERNAL_LIBS_DIR}/tbb/share)
set(TBB_ROOT_DIR ${EXTERNAL_LIBS_DIR}/tbb)
set(TBB_LIBRARY ${EXTERNAL_LIBS_DIR}/tbb/lib)
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
unset(TBB_ROOT_DIR)
unset(TBB_LIBRARY)
