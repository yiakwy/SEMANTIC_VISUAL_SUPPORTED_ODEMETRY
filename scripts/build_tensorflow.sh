#!/bin/bash
# Author: Lei Wang
# Date: Oct 2019
# Updated: March 27, 2020

set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENDOR_ROOT="${ROOT}/vendors/github.com"

# "loading libraries"
source "${ROOT}/scripts/utils.sh"

# comment utility
[ -z ${BASH} ] || shopt -s expand_aliases
alias COMMEND_BEGIN="if [  ]; then"
alias COMMEND_END="fi"

function Ubuntu_()
{
  # install build essentials
  info "Installing build essentials ..."
  sudo apt-get install -y \
    autoconf \
    automake \
    libtool \
    curl
    # make

  info "Installing tensorflow build dependencies ..."
  # install bazel
  # see details from https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu
  sudo apt-get install -y \
    pkg-config \
    g++ \
    zlib1g-dev \
    zip \
    unzip
 
  sudo apt install openjdk-11-jdk

  if which bazel > /dev/null; then
     info "bazel installed"
  else 
     info "Installing bazel ..."
     BAZEL_INSTALLER="bazel-2.2.0-installer-linux-x86_64.sh"
     cd ${ROOT}/scripts
     if [ ! -f ${BAZEL_INSTALLER} ]; then
     wget https://github.com/bazelbuild/bazel/releases/download/2.2.0/${BAZEL_INSTALLER} \
       -O ${BAZEL_INSTALLER}
     fi
     chmod +x ${BAZEL_INSTALLER}
     ./${BAZEL_INSTALLER} --user
     cd ${ROOT}
  fi
}

# install bazel
Ubuntu_ 

export PATH=$HOME/bin:$PATH

# build tensorflow_cc
if [ ! -d ${VENDOR_ROOT}/tensorflow_cc ]; then
git clone https://github.com/FloopCZ/tensorflow_cc.git "${VENDOR_ROOT}/tensorflow_cc"
fi
cd ${VENDOR_ROOT}/tensorflow_cc/tensorflow_cc
mkdir -p build && cd build
cmake ..
make 
sudo make install

COMMEND_BEGIN

TENSORFLOW_SOURCE_DIR=${VENDOR_ROOT}/tensorflow
TENSORFLOW_BUILD_DIR=${TENSORFLOW_SOURCE_DIR}/tensorflow_dist

# clone the vendor project to the source dir
if [ ! -d ${TENSORFLOW_SOURCE_DIR} ]; then
 mkdir -p ${TENSORFLOW_SOURCE_DIR}
 repo="https://github.com/tensorflow/tensorflow"
 git clone ${repo} ${TENSORFLOW_SOURCE_DIR}  
fi

# build
cd ${TENSORFLOW_SOURCE_DIR}
./configure --prefix=${TENSORFLOW_BUILD_DIR}
bazel build tensorflow:libtensorflow_gpu_all.so

# The following codes deprecated in March, 2020 in favor of new implementation of tensorflow_cc
# install
mkdir -p ${TENSORFLOW}
cp ${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow/*.so ${TENSORFLOW_BUILD_DIR}/lib
cp ${TENSORFLOW_SOURCE_DIR}/bazel-genfiles/tensorflow/core/framework/*.h ${TENSORFLOW_BUILD_DIR}/include/tensorflow/core/framework
cp ${TENSORFLOW_SOURCE_DIR}/bazel-genfiles/tensorflow/core/kernels/*.h   ${TENSORFLOW_BUILD_DIR}/include/tensorflow/core/kernels
cp ${TENSORFLOW_SOURCE_DIR}/bazel-genfiles/tensorflow/core/lib/core/*.h  ${TENSORLFOW_BUILD_DIR}/include/tensorflow/core/lib/core
cp ${TENSORFLOW_SOURCE_DIR}/bazel-genfiles/tensorflow/core/protobuf/*.h  ${TENSORFLOW_BUILD_DIR}/include/tensorflow/core/protobuf
cp ${TENSORFLOW_SOURCE_DIR}/bazel-genfiles/tensorflow/core/util/*.h      ${TENSORFLOW_BUILD_DIR}/include/tensorflow/core/util
cp ${TENSORFLOW_SOURCE_DIR}/bazel-genfiles/tensorflow/cc/ops/*.h         ${TENSORFLOW_BUILD_DIR}/include/tensorflow/cc/ops/

# copy third party libraries
cp -r ${TENSORFLOW_SOURCE_DIR}/third_party/ ${TENSORFLOW_BUILD_DIR}/
rm ${TENSORFLOW_BUILD_DIR}/third_party/py

COMMEND_END

