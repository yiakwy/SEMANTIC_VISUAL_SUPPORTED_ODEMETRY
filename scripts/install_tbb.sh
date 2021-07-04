#!/usr/bin/env bash
set -ex

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# import libs
source ${ROOT}/scripts/utils.sh

VERSION="2021.2.0"
PKG_NAME=tbb # intel tbb
PKG="v${VERSION}"
PKG_SRC=${PKG}.tar.gz

DOWNLOAD_LINK="https://github.com/oneapi-src/oneTBB/archive/refs/tags/${PKG_SRC}"

# Install system-provided intel tbb
# apt-get -y update && \
#   apt-get -y install \
#   libtbb-dev
# exit 0
# if ldconfig -p | grep -q libtbb ; then
#     info "Found existing Intel tbb installation. Skipp re-installation."
#     exit 0
# fi


TARGET_ARCH="$(uname -m)"

THREAD_NUM=$(expr `nproc` / 2 - 1)

download_if_not_cached "${PKG_NAME}" "${VERSION}" "${DOWNLOAD_LINK}" "${PKG}" "oneTBB-${VERSION}"

pushd $VENDOR_DIR/${PKG_NAME}/
    mkdir -p build && cd build
    set +x
    cmake .. \
        -DCMAKE_INSTALL_PREFIX="${MAPPING_EXTERNAL_DIR}/${PKG_NAME}" \
        -DCMAKE_BUILD_TYPE=Release
    make -j${THREAD_NUM}
    sudo make install
popd

ldconfig

ok "Successfully installed tbb ${VERSION}"
