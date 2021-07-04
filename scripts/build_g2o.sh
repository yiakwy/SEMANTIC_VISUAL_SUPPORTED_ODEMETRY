#!/bin/bash
ROOT="$( cd "$( dirname ${BASH_SOURCE[0]} )/.." && pwd )"

# loading libraries
source "${ROOT}/scripts/utils.sh"

G2O_SOURCE_DIR=${ROOT}/vendors/github.com/g2o
PLATFORM=linux/deb
INSTALLER=apt

# computed path
INSTALL_SCRIPT=${BASH_SOURCE[0]}
BUILDER_SCRIPT_PREFIX=${ROOT}/scripts/thirdparties/${PLATFORM}/${INSTALLER}

Ubuntu_() {
  info "no system dependencies..."
}

build_g2o() {
    # install the software
    info "installing g2o"

    VENDOR_DIR=${ROOT}/vendors/github.com/g2o

    pushd ${VENDOR_DIR}
    rm -rf build
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j${num_cores}
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

    build_g2o

  else
    warn "platform <$(uname -s)> Not supported yet. Please install the package dependencies manually. Pull requests are welcome!"
  fi
}

main
