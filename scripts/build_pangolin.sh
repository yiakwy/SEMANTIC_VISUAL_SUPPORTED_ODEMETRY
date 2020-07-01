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

# install depandencies
info "installing dependancies of ES2"
bash ${BUILDER_SCRIPT_PREFIX}/install_gles.sh
sudo python -mpip install numpy pyopengl Pillow pybind11
sudo apt install pkg-config
sudo apt install libegl1-mesa-dev libwayland-dev libxkbcommon-dev
wayland-protocols

# optional input for 

# install the software
info "installing pangolin"

VENDER_DIR=${ROOT}/vendors/github.com/Pangolin

cd ${VENDER_DIR}
rm -rf build
mkdir -p build
cd build
cmake ..
cmake --build .
cd -1 
