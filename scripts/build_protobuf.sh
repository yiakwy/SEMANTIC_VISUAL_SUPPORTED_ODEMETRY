#!/bin/bash
set -e

ROOT="$( cd "$( dirname ${BASH_SOURCE[0]} )/.." && pwd )"
VENDOR_ROOT="${ROOT}/vendors/github.com"

# loading libraries
source "${ROOT}/scripts/utils.sh"

# changed from 3.11.1 -> 3.0.0 to meet tensorflow request
VERSION="3.8.0"
if [[ ${VERSION} != "3.0.0" ]]; then
  PROTOBUF_PKG_DIR="https://github.com/protocolbuffers/protobuf/releases/download/v${VERSION}/protobuf-all-${VERSION}.tar.gz"

  cd ${VENDOR_ROOT}
  if [ ! -f protobuf-all-${VERSION}.tar.gz ]; then
    wget -P ${VENDOR_ROOT} ${PROTOBUF_PKG_DIR}
    tar -zxvf protobuf-all-${VERSION}.tar.gz
    mv protobuf-${VERSION} protobuf
  fi
  cd ${ROOT}
else 
  PROTOBUF_PKG_DIR="https://github.com/protocolbuffers/protobuf/archive/v${VERSION}.tar.gz"

  cd ${VENDOR_ROOT}
  if [ ! -f v${VERSION}-tar.gz ]; then
    wget -P ${VENDOR_ROOT} ${PROTOBUF_PKG_DIR}
    tar -zxvf v${VERSION}.tar.gz
    mv v${VERSION} protobuf
  fi
  cd ${ROOT}

fi

# install the software
info "installing protobuffer"
VENDOR_DIR=${ROOT}/vendors/github.com/protobuf

num_cores=`expr $(grep -c ^processor /proc/cpuinfo)`
if (( $num_cores > 32 )); then
  num_cores=32
fi

cd ${VENDOR_DIR}
./configure
make -j${num_cores}
make check
make test
sudo make install
sudo ldconfig
