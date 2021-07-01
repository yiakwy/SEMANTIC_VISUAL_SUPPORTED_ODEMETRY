#!/usr/bin/env bash
set -e

# computing absolute root
echo "$BASH_SOURCE"
ROOT="$( cd "$( dirname "$BASH_SOURCE[0]" )/../../" && pwd )"
echo "ROOT: $ROOT"

if [ -z "${DOCKER_REPO}" ];then
  DOCKER_REPO=openhdmap/study/svso
fi

TIME=""
ARCH=
TAG=
USE_PROXY="FALSE"
DOCKER_FILE=

Ubuntu_() {
  TIME=$(date +%Y%m%d_%H%M)
  ARCH=$(uname -m)
  TAG="dev-${ARCH}-${TIME}_$1"
}

build_img() {
  tag_suffix=$2
  Ubuntu_ $tag_suffix

  # set network proxy
  # if you are experiencing a slow network, consider arg UBUNTU_MIRROR and set it to the nearest endpoint.
  HTTP_PROXY="http://127.0.0.1:8888"
  HTTPS_PROXY="https://127.0.0.1:8888"
  args=(
  "--build-arg http_proxy=$HTTP_PROXY"
  "--build-arg https_proxy=$HTTPS_PROXY"
  )

  DOCKER_FILE=$1

  # to avoid bad influences from intermediate layers
  # remove dangling images : docker rmi -f $(docker images -f "dangling=true" -q)
  if [ USE_PROXY = "YES" ]; then
    echo "Using proxy HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY"
    docker build -t "$DOCKER_REPO:$TAG" \
      -f "$ROOT/docker/$DOCKER_FILE" \
      "$ROOT" $args
  else
    docker build -t "$DOCKER_REPO:$TAG" \
      -f "$ROOT/docker/$DOCKER_FILE" \
      "$ROOT"
  fi

}

# @todo TODO
update_img() {
 echo "Not Implemented Yet!"
}

main() {
# do not modify the lines below  unless you understand what you are doing ...

build_img "Dockerfile.base" "base"
# build_img "Dockerfile.devel.ml" "stage_ml"
}

main
