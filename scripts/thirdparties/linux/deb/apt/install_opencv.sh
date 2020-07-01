#!/bin/bash

# set -e

if [ ! -d ${PROJECT_ROOT} ]; then 
ROOT=${PROJECT_ROOT}
else
  # compute aboslute path whenver the script is executed
  ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../../../" && pwd)"
fi
REPO_HOME="github.com"
VENDOR_ROOT=${ROOT}/vendors/${REPO_HOME}

# this scripts will install opencv-4.1.0 (latest version)
sudo apt update
sudo apt install -y \
 python-dev \
 python3-dev \
 python-numpy \
 python3-numpy

# do not install opencv using conda for opencv-4.1.0, instead
# build it from source

declare -a pkgs=(
# opencv4 use qt-5 (linux or macos) or gtk (linux)
"libgtk-3-dev" 
)

INSTALLER=apt-get

# install dependancies
for pkg in ${pkgs[@]}; do
  sudo $INSTALLER install ${pkg}
done

# compile cmake-3.9 with https support, note native cmake from Ubuntu 18.04
# does not support it
num_cores=`expr $(grep -c ^processor /proc/cpuinfo)`
if (( $num_cores > 32 )); then
num_cores=32
fi

# Native cmake does not compile with https support (curl with ssl), hence
# we need to rebuild cmake
cmake_pkg_source="https://cmake.org/files/v3.9/cmake-3.9.0.tar.gz"
if [ ! -f ${VENDOR_ROOT}/cmake-3.9.0.tar.gz ]; then
wget --no-check-certificate ${cmake_pkg_source} -O ${VENDOR_ROOT}/

cd ${VENDOR_ROOT}
tar -zxvf cmake-3.9.0.tar.gz
mv cmake-3.9.0 cmake
cd cmake
./bootstrap --system-curl
make -j$num_cores && sudo make install
cd ${ROOT}
fi

# build opencv
cd ${VENDOR_ROOT}/opencv
# rm -r build
mkdir -p build
cd build

# see https://www.learnopencv.com/install-opencv-4-on-ubuntu-18-04/ with
# attention that config for python3 is incomplete. I have fixed it by reading
# CMakeLists.txt carefully
#
# also see discussion: https://github.com/opencv/opencv/issues/8039

args=(
"-DCMAKE_BUILD_TYPE=RELEASE"
"-DCMAKE_INSTALL_PREFIX=/usr/local"
"-DENABLE_FAST_MATH=ON"
"-DINSTALL_C_EXAMPLES=ON"
"-DINSTALL_PYTHON_EXAMPLES=ON"
"-DOPENCV_GENERATE_PKGCONFIG=ON"
"-DOPENCV_ENABLE_NONFREE=ON"
"-DOPENCV_EXTRA_MODULES_PATH=${VENDOR_ROOT}/opencv_contrib/modules"
"-DWITH_OPENGL=ON"
# PYTHON 3 BUNDLE SUPPORT
"-DBUILD_opencv_python3=ON"
"-DPYTHON3_EXECUTABLE=$(which python)"
"-DPYTHON_DEFAULT_EXECUTABLE=$(which python)"
"-DHAVE_opencv_python3=ON"
"-DPYTHON3_INCLUDE_DIR=$(python -c 'from distutils.sysconfig import \
get_python_inc; print(get_python_inc())')"
"-DPYTHON3_LIBRARIES=$(python -c 'from distutils.sysconfig import \
get_python_lib; print(get_python_lib())')"
"-DPYTHON3_NUMPY_INCLUDE_DIRS=$(python -c "import numpy as \
np;print(np.get_include())")"
"-DBUILD_EXAMPLES=ON"
)

cmake .. "${args[@]}"
make -j${num_cores}

# setup cv python
source setup_vars.sh

# check installation
pkg-config --modversion opencv4

