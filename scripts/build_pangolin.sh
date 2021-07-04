#!/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# loading libraries
source "${ROOT}/scripts/utils.sh"

PANGOLIN_SOURCE_DIR=${ROOT}/vendors/github.com/Pangolin
PLATFORM=linux/deb
INSTALLER=apt

# computed path
INSTALL_SCRIPT=${BASH_SOURCE[0]}
BUILDER_SCRIPT_PREFIX=${ROOT}/scripts/thirdparties/${PLATFORM}/${INSTALLER}

Ubuntu_() {
  # install depandencies
  info "installing dependancies of ES2"
  bash ${BUILDER_SCRIPT_PREFIX}/install_gles.sh
  sudo python -mpip install numpy pyopengl Pillow pybind11
  sudo apt install pkg-config
  sudo apt install libegl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols
}

build_pangolin() {
  # install the software
  info "installing pangolin"

  VENDER_DIR=${ROOT}/vendors/github.com/Pangolin

  pushd ${VENDER_DIR}
  rm -rf build
  mkdir -p build && cd build
  cmake ..
  cmake --build .
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

    build_pangolin

  else
    warn "platform <$(uname -s)> Not supported yet. Please install the package dependencies manually. Pull requests are welcome!"
  fi
}

main