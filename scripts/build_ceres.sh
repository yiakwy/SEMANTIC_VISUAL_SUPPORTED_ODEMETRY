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

#
num_cores=

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

build_ceres_solver() {
# install the software
info "installing ceres solver"

VENDOR_DIR=${ROOT}/vendors/github.com/ceres-solver

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
}

main() {
	if [ $(uname -s) == "Darwin" ]; then
		OS="OSX"
    
        MAC_MAJOR_VERSION=$(sw_vers -productVersion | awk -F "." '{print $1 "." $2}')
        MAC_MINOR_VERSION=$(sw_vers -productVersion | awk -F "." '{print $3}')      

		info "<$(uname -s)> detected. checking Max OSX versions: $MAC_MAJOR_VERSION.$MAC_MINOR_VERESION ..."
		info "installing ..."
		# OSX $MAC_MAJOR_VERSION

        PLATFORM=osx
        INSTALLER=brew

BUILDER_SCRIPT_PREFIX=${ROOT}/scripts/thirdparties/${PLATFORM}/${INSTALLER}

		#

        num_cores=$(sysctl -n hw.ncpu)

	else
		if [ -f /etc/lsb-release ]; then
		OS="Ubuntu"

        PLATFORM=linux/deb
        INSTALLER=apt

BUILDER_SCRIPT_PREFIX=${ROOT}/scripts/thirdparties/${PLATFORM}/${INSTALLER}

		Ubuntu_
        num_cores=`expr $(grep -c ^processor /proc/cpuinfo)`    

		else
		warn "platform <$(uname -s)> Not supported yet. Please install the package dependencies manually. Pull requests are welcome!"
		fi
	fi 

    build_ceres_solver

}

main

