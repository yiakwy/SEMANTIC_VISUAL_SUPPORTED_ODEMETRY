FROM openhdmap/study/svso:dev-x86_64-20200804_0134_base-v1

MAINTAINER LEI WANG

LABEL version="0.1.2"
LABEL description="build stage 1"

# https://github.com/phusion/baseimage-docker/issues/319
ENV DEBIAN_FRONTEND teletype

RUN /bin/bash -c "source ~/.bashrc" && cat ~/.bashrc

ENV PATH ~/anaconda3/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
RUN which conda && conda info --envs

RUN pip --version
RUN python --version

ADD ./docker/scripts/init_python.sh /tmp/deploy/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/docker/scripts/

WORKDIR /tmp/deploy/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY
RUN du -h --max-depth=1 docker/scripts

# build python dependancies
COPY ./python/requirements.txt /tmp/deploy/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/python/requirements.txt
RUN pip install --upgrade pip
RUN conda update --all -y 
RUN pip install --upgrade --default-timeout=100 -i https://pypi.tuna.tsinghua.edu.cn/simple \
	tensorflow-gpu==2.2.0 \
	protobuf==3.11.1

# https://github.com/tensorflow/tensorflow/issues/30191
# Download tensorflow binaries would take 2 hours and we put it ahead in case failure of building the docker image

ADD ./vendors/github.com/coco /tmp/deploy/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/vendors/github.com/coco
ADD ./vendors/github.com/Mask_RCNN /tmp/deploy/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/vendors/github.com/Mask_RCNN

RUN bash ./docker/scripts/init_python.sh

CMD ["nvidia-smi"]
