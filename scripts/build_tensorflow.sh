#!/bin/bash
# Author: Lei Wang
# Date: Oct 2019
# Updated: March 27, 2020

set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
VENDOR_ROOT="${ROOT}/vendors/github.com"

# downgrade bazel version from 2.2.0 to 2.0.0 build tensorflow 2.2.0
VERSION="2.0.0" # 2.2.0

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
     BAZEL_INSTALLER="bazel-${VERSION}-installer-linux-x86_64.sh"
     cd ${ROOT}/scripts
     if [ ! -f ${BAZEL_INSTALLER} ]; then
     wget https://github.com/bazelbuild/bazel/releases/download/$VERSION/${BAZEL_INSTALLER} \
       -O ${BAZEL_INSTALLER}
     fi
     chmod +x ${BAZEL_INSTALLER}
     ./${BAZEL_INSTALLER} --user
     cd ${ROOT}
  fi
}

# install bazel
Ubuntu_ 

# add bazel path
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
