#!/bin/bash

set -e

if [[ $EUID -ne 0 ]]; then
  echo "$0: must be root" 1>&2
  exit 1
fi

# get your ubuntu distribution
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list > \
      /etc/apt/sources.list.d/nvidia-docker.list

# nvidia-docker2 installs nvidia-container-cli and /etc/docker/daemon.json
apt-get update && apt-get install -y nvidia-docker2

# restart docker and load nvidia-docker2.
# Note for docker >= 19.03, you don't need nvidia-docker2 anymore, and no runtime for nvidia needed to build an image from dockerfile

# restart docker service to load nvidia-container-cli
service docker restart
