SEMANTIC_VISUAL_SUPPORTED_Odemetry
==================================

This is a sparse Monocular SLAM project merely for research in HDMap hibrid pipeline. Constrast to dense or recent popular semi-dense slam projects like "DTAM", "LSD-SLAM", "DVO", this is a implementation of Visual Odemetry \(VO\) using semantic feature extraction \(SFE\) of observations \(ROI\) for sparse depth graph registration 
and precise relocalization solutions.

![mapping](https://drive.google.com/file/d/10ZMLZB9dmMf8OUzE1wOK3-trtCnwHpA_/view?usp=sharing)

![relocalization](https://drive.google.com/file/d/1D04cuHaEC1v70PSUDf0K1AL9gbJxp2xb/view?usp=sharing)

The purpose of the project is by applying semantic segmentation and recoginiztion in videos while introducing spatial information of depth graph triangulated from key points, the project wants to show that:

1. Sparse SLAM like ORBSlam is mainly for odemetry or localization process. By introducing semantic information of observations from camera, we can construct valid mapping procedures using sparse estimated 3d points, though the accuracy may vary with various factors.

2. BoW for images is a method of clustering, but should not the ideal one; we can use natural segmented instances to group and recall key points, frames. In this project, we use Landmark to recall associated keypoints and frames for PnP process which is very easy to do. Our tests show that the computed pose are very close to ground truth.

3. In sparse SLAM, matching is accurate only when two frames are close enough. This holds even though we apply different strategies of test ratios with risks of greatly reducing mathing of points. 

But meanwhile, triangulation from initialization or local mapping statges would be greately improved when stereo disparity becomes large and more frames \(> 2 frames\)  are involved (Global Bundle Adjustment). This means two frames used in matching can't be close. 

Hence I made great efforts in designing tracking state machine and methods for matching to resolve above mentioned delimma to get very good mathching results but preserve triangulation precisions.

![matching](https://drive.google.com/file/d/1Wca-gyz4EzCQsOlwfMhVewVDFs-erDrb/view?usp=sharing)

## Architecture

The solution compromises tracking state machine using sparse keypoints and semantic detections both for localization
and sparse mapping. Contrast to merely using keppoints in sparse SLAM, semnatic detection and matching of objects,
will greatly boost matching performance and gives more accurate sparse mapping.

Hence we extend the concept of lanmarks and data structures of covisibility graph, which leads to recalling keypoints through
ROI instead of using precomputed Bag of Words method. 

One of the application of such semantic slam is relocalization method. I implemented a full functional Relocalization based on
Bipartite landmarks matching for initial alignment and ICP algorithm to compute transform of a robot.

![framework](https://drive.google.com/file/d/1UwCpduO2uADV8Pt_eZFWER-xOFNlaoll/view?usp=sharing)

## Installation

### Dependencies

The most of libraries and dependencies could be installed automatically by provided scripts in "$ROOT/scripts". But there are still some third party packages
needed be installed from source manually. Instructions or automation scripts provided. Third parties projects built from source will be distributed into "${ROOT}/vendors/${REPO}/". For example, we build 
tensorflow inside "${ROOT}/vendors/github.com/tensorflow_cc/tensorflow_cc/tensorflow".

### Env

The default system is built upon `Ubuntu 18.04` but other verions are also possible to be sopported. The hardware includes a physical GeForce RTX GPU (compute ability 7.5) and monocular camera with a workable scale advisor device. 

According to the NV website and our tests, `cuda 10.1`, `cudnn` 7.6.5 is just enough. At the beginning, I choose MASK-RCNN pre-trained with coco dataset for POC. You change the model to lighter one to compute semantic features for instances. In our tests, the features are not distinguwshable from objects with the same label, hence we use
**UIoU** I invented last year in ROI matching procesure \(see `ROIMatcher`\).  

### Step1: Build general dependencies of the project 

> bash scripts/install.sh

### Step2: Install anaconda conda and create virtual environments with python3.6 by conda

The above step will install c++ development libraries together with ros packages.

### Step3: Install python dependencies

This step creates environment to run python codes inside the project, which `including all in one` Semantic Visual Supported Odemetry \(SVOSO\) tracker
in "${ROOT}/notebooks/svso_tracker.ipynb" and python implementation of svso in "${ROOT}/python"

> bash scripts/init_python.sh

### Step4: Build Opencv4 libraries and python bindings

> bash scripts/thirdparty/linux/deb/apt/

The above step will automatically install build essentials, cmake with ssl support and opencv4 with python bindings to the current active python binary.

### Step5: C++ specific devleopment libraries

We moved our major backend into c++ version to make full use of concurrency and parallel computing abilities. The backend include 

- **a tracker** to estimate camera poses and extract landmarks for key points registration, key frame selection and keypoints depth triangluation procedures, 
- **local mapping** thread with ROI 
- and finally **a relocalization** thread to update map when a scenario change and visited objects detected.

##### Step 5-1: Install Protobuf

protobuf will be used by the cmake based project to build transportation layer of structurese a slam program and help in
 generating codes in c++ side. `Frame`, `Point3D`, `Pixel2D`, `RuntimeBlock` and etc. will be automatically generated by the programe.

> bash scripts/install_protoc.sh


##### Step 5-2: Install ceres, g2o and pangolin used for local mapping optimization

First fetch third parties project sources specified in ".gitmodules". You might encounter "git clone" problems using `git submodure init` command
. If that happens, download latest released sources `*.tar.gz` in whatever method that you feel good and extract the vendors location.

Then simple run

> bash scripts/build_ceres.sh
> bash scripts/build_g2o.sh
> bash scripts/build_pangolin.sh

Ceres relies on Eigen-3.3.3 where tensorflow 2.2.0-rc we are going to build uses patched eigen 3.3.9 mainted by
bazel system. To avoid conflicts I modified our `cmake/Modules/FindEigen3.cmake` to a version of bazel installed recorded in
`tensorflow/workspace.bzl` you can also find the library in `~/.cache/bazel/_bazel_${USER}/external/${HASH_CODE}/eigen_archive` [3]

##### Step 5-3: Build tensorflow c++ runtime from source code

Tensorflow is extremely large. First you need to build bazel then build shared library so that you can use in our cmake project.
Building Tensorfow with Bazel might takes at least 45 minutes \(tensorflow build time\) and up to 3 ~ 4 hours (bazel fetches dependencies).
The process is dependant on your network status.

> bash scipts/build_tensorflow.sh

If script does not work \(due to your network proxy, git ssh configuration, git cache setting and many other reasons\). Here are the procedures to do:

1) go to [tensorflow_cmake](https://github.com/cjweeks/tensorflow-cmake/blob/master/README.md) and follow the instructions. Note

Since bazel consumes a large portion of memories \[1\]\[2\] which could break your building process, replace bazel build command with the following one:

> sudo bazel build --jobs=8 --config=monolithic tensorflow:libtensorflow_all.so

## Data

All the data reside in "$ROOT/data/${DATASET\_SOURCE}/${DATASET\_NAME}". Curerntly I have tested TUM hand hold camera dataset with rgbd ground truths for references.
More dataset will be supported soon

## Build

Run

> mkdir -p build && cd build && cmake .. && make -j8

or using building script "${ROOT}/scripts/build.sh" where we provide flags control. The c++ implementation is automatically generated. other language protobuf implementation is generated by 

> bash scripts/gen_proto --lang=python

## References
```text
[1] github.com/tensorflow/tensorflow/issues/38183
[2] github.com/FloopCZ/tensorflow_cc/issues/213
[3] https://github.com/tensorflow/tensorflow/issues/38237
```
