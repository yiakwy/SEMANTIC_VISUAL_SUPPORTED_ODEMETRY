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
     # add bazel path
     export PATH=$HOME/bin:$PATH
  fi
}

build_tensorflow() {
  # install the software
  info "installing tensorflow"

  VENDER_DIR=${ROOT}/vendors/github.com/tensorflow_cc
  if [ ! -d ${VENDOR_DIR} ]; then
    git clone https://github.com/FloopCZ/tensorflow_cc.git "${VENDOR_DIR}"
  fi

  pushd ${VENDER_DIR}
  rm -rf build
  mkdir -p build && cd build
  cmake ..
  make # -j${num_cores}
  sudo make install
  popd
}

main() {
  if [ -f /etc/lsb-release ]; then
    OS="Ubuntu"

    PLATFORM=linux/deb
    INSTALLER=apt

    BUILDER_SCRIPT_PREFIX=${ROOT}/scripts/thirdparties/${PLATFORM}/${INSTALLER}

    Ubuntu_
    num_cores=`expr $(grep -c ^processor /proc/cpuinfo)`
    if (( $num_cores > 32 )); then
      num_cores=32
    fi

    build_tensorflow

  else
    warn "platform <$(uname -s)> Not supported yet. Please install the package dependencies manually. Pull requests are welcome!"
  fi
}

main
