  # Finds the required directories to include Eigen. Since Eigen is
# only header files, there is no library to locate, and therefore
# no *_LIBRARIES variable is set.

# I copied these utilites from cmake_modules/FindEigen3.cmake, credits to relevant developers (Lei/Yi)
if(NOT Eigen3_FIND_VERSION)
  if(NOT Eigen3_FIND_VERSION_MAJOR)
    set(Eigen3_FIND_VERSION_MAJOR 2)
  endif(NOT Eigen3_FIND_VERSION_MAJOR)
  if(NOT Eigen3_FIND_VERSION_MINOR)
    set(Eigen3_FIND_VERSION_MINOR 91)
  endif(NOT Eigen3_FIND_VERSION_MINOR)
  if(NOT Eigen3_FIND_VERSION_PATCH)
    set(Eigen3_FIND_VERSION_PATCH 0)
  endif(NOT Eigen3_FIND_VERSION_PATCH)

  set(Eigen3_FIND_VERSION "${Eigen3_FIND_VERSION_MAJOR}.${Eigen3_FIND_VERSION_MINOR}.${Eigen3_FIND_VERSION_PATCH}")
endif(NOT Eigen3_FIND_VERSION)

macro(_eigen3_check_version)
  file(READ "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" _eigen3_version_header)

  string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen3_world_version_match "${_eigen3_version_header}")
  set(EIGEN3_WORLD_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen3_major_version_match "${_eigen3_version_header}")
  set(EIGEN3_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen3_minor_version_match "${_eigen3_version_header}")
  set(EIGEN3_MINOR_VERSION "${CMAKE_MATCH_1}")

  set(EIGEN3_VERSION ${EIGEN3_WORLD_VERSION}.${EIGEN3_MAJOR_VERSION}.${EIGEN3_MINOR_VERSION})
  if(${EIGEN3_VERSION} VERSION_LESS ${Eigen3_FIND_VERSION})
    set(EIGEN3_VERSION_OK FALSE)
  else(${EIGEN3_VERSION} VERSION_LESS ${Eigen3_FIND_VERSION})
    set(EIGEN3_VERSION_OK TRUE)
  endif(${EIGEN3_VERSION} VERSION_LESS ${Eigen3_FIND_VERSION})

  if(NOT EIGEN3_VERSION_OK)

    message(STATUS "Eigen3 version ${EIGEN3_VERSION} found in ${EIGEN3_INCLUDE_DIR}, "
                   "but at least version ${Eigen3_FIND_VERSION} is required")
  endif(NOT EIGEN3_VERSION_OK)
endmacro(_eigen3_check_version)

# set EIGEN3_INCLUDE_DIR
# this is where I installed bazel eigen refered by tensorflow 2.2.0rc
unset(EIGEN3_FOUND)
# /usr/include/eigen3, ceres depended eigen 
# /usr/local/include/eigen3/, extracted from bazel archive
# set(EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3")
set (EIGEN3_INCLUDE_DIR "/home/yiak/.cache/bazel/_bazel_yiak/3a9860bf2dd6115a1f3a2f621e74b511/external/eigen_archive")
_eigen3_check_version()
set(EIGEN3_FOUND ${EIGEN3_VERSION_OK})

include(FindPackageHandleStandardArgs)
echo ("EIGEN3_VERSION: ${EIGEN3_VERSION}")

echo ("Eigen3_DIR: ${Eigen3_DIR}")
find_path(EIGEN3_INCLUDE_DIR
        NAMES
        unsupported
        Eigen
        signature_of_eigen3_matrix_library
        HINTS
        ${EIGEN_INSTALL}
        # ${EIGEN3_INCLUDE_DIR}
        )

# set Eigen_FOUND
find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR)

# set external variables for usage in CMakeLists.txt
if(EIGEN3_FOUND)
  echo ("Found Eigen3, inc: ${EIGEN3_INCLUDE_DIR}")
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

# hide locals from GUI
mark_as_advanced(EIGEN3_INCLUDE_DIR)

