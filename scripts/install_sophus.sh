#!/usr/bin/env bash
set -ex

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# import libs
source ${ROOT}/scripts/utils.sh

VERSION="1.0.0"
PKG_NAME=Sophus
PKG="${PKG_NAME}-${VERSION}"
PKG_SRC=v${VERSION}.tar.gz

DOWNLOAD_LINK="https://github.com/strasdat/Sophus/archive/${PKG_SRC}"

TARGET_ARCH="$(uname -m)"

THREAD_NUM=$(expr `nproc` / 2 - 1)

download_if_not_cached "${PKG_NAME}" "${VERSION}" "${DOWNLOAD_LINK}" ${PKG}

pushd $VENDOR_DIR/${PKG_NAME}/
    mkdir -p build && cd build
    set +x
    cmake .. \
        -DYAML_BUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="${MAPPING_EXTERNAL_DIR}/${PKG_NAME}" \
        -DCMAKE_BUILD_TYPE=Release
    make -j${THREAD_NUM}
    sudo make install
popd

ldconfig

ok "Successfully installed ${PKG_NAME} ${VERSION}"
