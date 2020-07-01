#!/bin/bash
ROOT="$( cd "$( dirname ${BASH_SOURCE[0]} )/.." && pwd )"

# loading libraries
source "${ROOT}/scripts/utils.sh"

CERES_SOURCE_DIR=${ROOT}/vendors/github.com/ceres-solver
PLATFORM=linux/deb
INSTALLER=apt

# computed path
INSTALL_SCRIPT=${BASH_SOURCE[0]}
BUILDER_SCRIPT_PREFIX=${ROOT}/scripts/thirdparties/${PLATFORM}/${INSTALLER}


# install dependancies
function Ubuntu_() 
{
sudo apt-get install -y \
  libatlas-base-dev \
  libeigen3-dev

# see ceres installation home page
sudo add-apt-repository ppa:bzindovic/suitesparse-bugfix-1319687
sudo apt-get install -y libsuitesparse-dev 
}

Ubuntu_

# install the software
info "installing ceres solver"

VENDOR_DIR=${ROOT}/vendors/github.com/ceres-solver

num_cores=`expr $(grep -c ^processor /proc/cpuinfo)`
if (( $num_cores > 32 )); then
  num_cores=32
fi

cd ${VENDOR_DIR}
rm -rf build
mkdir -p build
cd build
cmake ..
make -j${num_cores}
make test
sudo make install
