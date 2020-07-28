#!/bin/bash
ROOT="$( cd "$( dirname ${BASH_SOURCE[0]} )/.." && pwd )"

# "loading libraries"
source "${ROOT}/scripts/utils.sh"

VENDOR_DIR="${ROOT}/vendors/github.com"
VERSION="3.11.1"

function Ubuntu_()
{
  # install build essentials
  info "Installing gnu build essentials"
  sudo apt-get install -y \
    autoconf \
    automake \
    libtool  \
    curl
    
  # recall that  we have install make manually to enable https for cmake
  
  num_cores=`expr $(grep -c ^processor /proc/cpuinfo)`
  if (( $num_cores > 32 )); then
  num_cores=32
  fi

  if which protoc > /dev/null && [ $(protoc --version | head -n1 | cut -d" " -f4) -lt $VERSION ]; then
    info "protoc has been installed"
  else
    info "installing protoc ..."
    PROTOC_PKG="https://github.com/protocolbuffers/protobuf/releases/download/v$VERSION/protobuf-all-$VERSION.tar.gz"
    if [ ! -f ${VENDOR_DIR}/protobuf-all-$VERSION.tar.gz ]; then
    wget $PROTOC_PKG -P ${VENDOR_DIR}
    fi
    cd ${VENDOR_DIR}
    tar -zxvf protobuf-all-$VERSION.tar.gz
    mv protobuf-$VERSION protobuf
    cd protobuf
    ./configure
    make -j${num_cores}
    make check
    sudo make install 
    sudo ldconfig # refresh shared library cache
  fi
}

Ubuntu_
