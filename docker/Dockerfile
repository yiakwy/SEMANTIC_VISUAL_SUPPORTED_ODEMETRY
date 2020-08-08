FROM nvidia/cuda:10.2-base-ubuntu18.04

MAINTAINER LEI WANG <yiak.wy@gmail.com>

LABEL version="0.1.2"
LABEL description=""

# https://github.com/phusion/baseimage-docker/issues/319
ENV DEBIAN_FRONTEND teletype

# project referred from context
ADD . /tmp/deploy/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY

# install general dependencies
RUN ls -h
RUN ls -h /tmp/deploy/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY

WORKDIR /tmp/deploy/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY
RUN du -h --max-depth=1
# remove nvidia repo from source
RUN mv /etc/apt/sources.list.d/cuda.list /tmp/cuda.list.bak && mv /etc/apt/sources.list.d/nvidia-ml.list /tmp/nvidia-ml.list.bak

# uncomment it for network test
# RUN bash ./docker/scripts/test_network_connectivity.sh

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get -y --no-install-recommends install \
 sudo \
 apt-utils

RUN DEBIAN_FRONTEND=noninteractive bash ./scripts/thirdparties/linux/deb/apt/prepare_ubuntu.sh
RUN DEBIAN_FRONTEND=noninteractive bash ./scripts/install.sh

# Note before Docker 19.03, one has to install nvidia-driver in the container

# step 1: Install anaconda for python 3.6
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc && \
    echo "export PATH=/root/anaconda3/bin:$PATH" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc " && \
    cat ~/.bashrc

# https://github.com/ContinuumIO/docker-images/issues/136
# SHELL ["/bin/bash", "-l", "-c"]

ENV PATH ~/anaconda3/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
RUN which conda && conda info --envs

RUN pip --version
RUN python --version

CMD ["/bin/bash", "nvidia-smi"]

