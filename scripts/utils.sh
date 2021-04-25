set -e

COLOR_OFF='\033[0m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'

function info() {
	(echo >&2 -e "${GREEN} [INFO] ${COLOR_OFF} $*")
}

function err() {
	(echo >&2 -e "${RED} [ERROR] ${COLOR_OFF} $*" )
}

function warn() {
	(echo >&2 -e "${YELLOW} [WARNING] ${COLOR_OFF} $*" )
}

function ok() {
  (echo >&2 -e "${GREEN}${BOLD} [ok] ${COLLOR_OFF} $*" )
}

ROOT="$( cd "$( dirname ${BASH_SOURCE[0]} )/.." && pwd )"
REPO="github.com"
VENDOR_DIR="${ROOT}/vendors/${REPO}"
MAPPING_EXTERNAL_DIR="$HOME/mapping_external"

mkdir -p $MAPPING_EXTERNAL_DIR/{bin,include,lib,share}

if [[ "$(uname -m)" == "x86_64" ]]; then
    export SUPPORTED_NVIDIA_SMS="5.2 6.0 6.1 7.0 7.5 8.0 8.6"
else # AArch64
    export SUPPORTED_NVIDIA_SMS="5.3 6.2 7.2"
fi

function download_if_not_cached() {
  local pkg_name="$1"
  local pkg_ver="$2"
  local pkg=$pkg_name-$pkg_ver
  local pkg_src=$pkg.tar.gz
  local pkg_repo="$3"

  pushd $VENDOR_DIR  
    if [ ! -f ${pkg_name} ]; then
       if [ ! -f ${pkg_src} ]; then
         wget -P ${VENDOR_DIR} ${pkg_repo}
       fi
       tar -xvf ${pkg_src}
       mv ${pkg} ${pkg_name}
    else
       info "$pkg_name exits"
       if [ ! -f ${pkg_src} ]; then
          if [ -d ${pkg_name}/.git ]; then
             warn "please checkout to ${pkg_ver} branch"
          else
             error "$pkg_name is not supported repo!"
          fi
       fi
    fi 
  popd
}
