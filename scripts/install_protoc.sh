#!/bin/bash
ROOT="$( cd "$( dirname ${BASH_SOURCE[0]} )/.." && pwd )"

# "loading libraries"
source "${ROOT}/scripts/utils.sh"

VENDOR_DIR="${ROOT}/vendor/github.com"

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

  if which protoc > /dev/null; then
    info "protoc has been installed"
  else
    info "installing protoc ..."
    PROTOC_PKG="https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz"
    wget $PROTOC_PKG -P ${VENDOR_DIR}
    cd ${VENDOR_DIR}
    tar -zxvf protobuf-all-3.6.1.tar
    mv protobuf-all-3.6.1.tar protobuf
    cd protobuf
    ./configure
    make -j${num_cores}
    make check
    sudo make install 
    sudo ldconfig # refresh shared library cache
  fi
}

Ubuntu_
