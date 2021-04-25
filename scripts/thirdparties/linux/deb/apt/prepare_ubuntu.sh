# install build essentials in case that we compile and build gnu software
sudo apt-get update && apt-get install -y \
	build-essential \
	llvm \
	git \
	vim \
	curl \
	aptitude \
	libatlas-base-dev \
	libboost-all-dev \
	libconsole-bridge-dev \
	libcurl4-openssl-dev \
	libgflags-dev \
	libflann-dev \
	libfreetype6-dev \
	libgoogle-glog-dev \
	libhdf5-serial-dev \
	libicu-dev \
	libleveldb-dev \
	liblz4-dev \
	liblmdb-dev \
	libopencv-dev \
	libopenni-dev \
	libpoco-dev \
	libproj-dev \
	libpython2.7-dev \
	libqhull-dev \
	libsnappy-dev \
	libtinyxml-dev \
	libyaml-cpp-dev \
	libyaml-dev \
	mpi-default-dev \
	python-matplotlib \
	python-pip \
	python-virtualenv \
	python-scipy \
	manpages-pl \
	software-properties-common \
	tmux \
	zip \
	unzip \
	wget \
	libaio-dev # see Linux libaio and io_uring

# !important
sudo ln -s /usr/lib/python2.7/dist-packages/vtk/libvtkRenderingPythonTkWidgets.x86_64-linux-gnu.so /usr/lib/x86_64-linux-gnu/libvtkRenderingPythonTkWidgets.so
sudo ln -s /usr/bin/vtk6 /usr/bin/vtk
