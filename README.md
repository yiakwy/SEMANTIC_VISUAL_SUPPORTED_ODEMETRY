Semantic Visual Supported Odemetry
==================================

This is a sparse Monocular SLAM project merely for research in the realm of HDMap hibrid pipeline construction. Constrast to dense or recent popular semi-dense slam projects such as "DTAM", "LSD-SLAM", "DSO", this is an implementation of Visual Odemetry \(VO\) using semantic feature extractor \(SFE\) of observations \(i.e., ROI\) for sparse depth graph registration and precise relocalization solutions.

<img src="https://drive.google.com/uc?export=view&id=10ZMLZB9dmMf8OUzE1wOK3-trtCnwHpA_" 
alt="mapping" width="449" height="180" border="10" />

<img src="https://drive.google.com/uc?export=view&id=1D04cuHaEC1v70PSUDf0K1AL9gbJxp2xb"
alt="relocalization" width="240" height="361" border="10" />

The purpose of the project is that by applying semantic segmentation and recoginiztion in videos while introducing spatial information of depth graph triangulated from key points, the project wants to show that:

1. Sparse SLAM like ORBSlam which is mainly for odemetry or localization process. By introducing semantic information of observations from camera, we can construct valid mapping procedures using sparse estimated 3d points, though the accuracy may vary with various factors.

2. BoW for images is a method of clustering, but should not be the ideal one; we can use naturally segmented instances to group and recall key points, frames and mapblocks. In this project, we use Landmark to recall associated keypoints and frames for PnP process which is very easy to do. Our tests show that the computed poses are very close to ground truth.

3. In sparse SLAM, matching is accurate only when two frames are close enough. This holds even though we apply different strategies of test ratios with risks of greatly reducing mathing of points  
  
   But meanwhile, triangulation from initialization or local mapping statges would be greately improved when stereo disparity becomes large and more frames \(> 2 frames\)  are involved (Global Bundle Adjustment). This means that two frames used in matching can't be close.  
Hence I made great efforts in designing tracking state machine and methods for matching to resolve above mentioned delimma to get very good mathching results while preserve triangulation precisions.

<img src="https://drive.google.com/uc?export=view&id=1Wca-gyz4EzCQsOlwfMhVewVDFs-erDrb" 
alt="matching" width="240" height="180" border="10" />

## Architecture

The solution compromises tracking state machine using sparse keypoints and semantic detections both for localization
and sparse mapping. Contrast to merely using keppoints in sparse SLAM, semantic detection and matching of those objects,
will greatly boost keypoints matching performance and give more accurate sparse mapping.

Hence we extend the concept of lanmarks and data structures of covisibility graph, which leads to recalling keypoints through
ROI instead of using precomputed Bag of Words method. 

One of the application of such semantic SLAM is the relocalization method. I implemented a full functional Relocalization based on
Bipartite landmarks matching for initial alignment and ICP algorithm to compute transform of a robot.

<img src="https://drive.google.com/uc?export=view&id=1UwCpduO2uADV8Pt_eZFWER-xOFNlaoll"
alt="framework" width="240" height="180" border="10" />

More technical details about data stucture, system design and algorithms can be found at the public report [SVSO](https://drive.google.com/uc?export=view&id=1XWf_esVG2gso-aZVtplwyTEWV8a8iSFa).

## Install and Debug From Docker

see "docs/install_and_debug_from_docker.md"

## Install From Source

#### Dependencies

The most of libraries and dependencies could be installed automatically by provided scripts in "$ROOT/scripts". But there are still some third party packages
needed be installed from source manually. Instructions or automation scripts provided. Third party projects built from source will be distributed into "${ROOT}/vendors/${REPO}/". For example, we build 
tensorflow inside "${ROOT}/vendors/github.com/tensorflow_cc/tensorflow_cc/tensorflow".

#### Env

The default system is built upon `Ubuntu 18.04` but other verions are also possible to be sopported. The hardware includes a physical GeForce RTX GPU (compute ability 7.5) and monocular camera with a workable scale advisor device. 

According to the NV website and our tests, `cuda 10.2`, `cudnn` 7.6.5 is just enough. At the beginning, I choose MASK-RCNN pre-trained with coco dataset for POC. You can change the model to lighter one to compute semantic features for instances. In our tests, the features are not distinguwshable from objects with the same label, hence we use
**UIoU** I invented last year in ROI matching procesure \(see `ROIMatcher`\) to keep track of identified objects.  

#### Step1: Build general dependencies of the project 

> bash scripts/install.sh

#### Step2: Install anaconda conda and create virtual environments with python3.6 by conda

The above step will install c++ development libraries together with ros packages.

#### Step3: Install python dependencies

This step creates environment to run python codes inside the project, which including experimental `all in one`\(out of date\) Semantic Visual Supported Odemetry \(SVOSO\) tracker
in "${ROOT}/notebooks/svso_tracker.ipynb" and python implementation of svso in "${ROOT}/python"

> bash scripts/init_python.sh

#### Step4: Build Opencv4 libraries and python bindings

> bash scripts/thirdparty/linux/deb/apt/install\_opencv.sh

The above step will automatically install build essentials, cmake with ssl support and opencv4 with python bindings to the current active python binary.

Once built successful, then check wether "cv2.so" resides in python installation directory computed by cmake:

```
# Ubuntu 18.04
(py36) ➜  apt git:(master) ✗ ls /usr/local/lib/python3.6/site-packages/cv2/python-3.6/    
cv2.cpython-36m-x86_64-linux-gnu.so  cv2.so

```

Make it visiable by anaconda python:

```
# create soft links to target installation of opencv 4 python bindings
ln -s /usr/local/lib/python3.6/site-packages/cv2  /home/$USER/anaconda3/envs/py36/lib/python3.6/site-packages/cv2

# rename the shared library
# OS_SUFFIX:
#  linux : gnu
#  macos : darwin
cd /home/$USER/anaconda3/envs/py36/lib/python3.6/site-packages/cv2/python-3.6
mv cv2.cpython-36m-x86_64-linux-${OS\_SUFFIX}.so cv2.so
```

#### Step5: C++ specific devleopment libraries

We moved our major backend into c++ version to make full use of concurrency and parallel computing abilities. The backend includes

- **a tracker** to estimate camera poses and extract landmarks for key points registration, key frame selection and depth triangluation procedures, 
- **local mapping** thread with ROI 
- and finally **a relocalization** thread to update map when a scenario change and visited objects detected.

##### Step 5-1: Install Protobuf

protobuf will be used by the cmake based project to build transportation layer of structurese a slam program and help in
 generating codes in c++ side. `Frame`, `Point3D`, `Pixel2D`, `RuntimeBlock` and etc. will be automatically generated by the programe.

> bash scripts/install_protoc.sh

##### Step 5-2: Install ceres, g2o and pangolin used for local mapping optimization

First fetch sources of third party projects specified in ".gitmodules". You might encounter "git clone" problems using `git submodure init` command
. If that happens, download latest released sources `*.tar.gz` in whatever method that you feel good and extract them to the vendors location.

After add git submoduels, 

```
git submodule add -f ${Pangolin_Git_Repo} vendors/github.com/pangolin
git submodule add -f ${Ceres_Solver_Git_Repo} vendors/github.com/ceres-solver
# note we not goint to include opence as submodules since it is too large and we have many problems to clone it directly
```

then simple run

> bash scripts/build_ceres.sh (Optional)
> bash scripts/build_g2o.sh
> bash scripts/build_g2opy.sh (Generate py bindings to be used in python)
> bash scripts/build_pangolin.sh

Ceres relies on Eigen-3.3.3 where tensorflow 2.2.0-rc we are going to build uses patched eigen 3.3.9 mainted by
bazel system. To avoid conflicts I modified our `cmake/Modules/FindEigen3.cmake` to a version of bazel installed recorded in
`tensorflow/workspace.bzl` you can also find the library in `~/.cache/bazel/_bazel_${USER}/external/${HASH_CODE}/eigen_archive` [3]

Also make sure that no eigen3 copies installed in the path of higher priority than the above path : "/usr/include/eigen3", "/usr/include/eigen3"

##### Step 5-3: Build tensorflow c++ runtime (v2.2.0-rc)  from source code

Tensorflow is extremely large. First you need to build bazel then build shared library so that you can use it in our cmake project.
Building Tensorflow with Bazel might take at least 45 minutes \(tensorflow build time\) and up to 3 ~ 4 hours (bazel fetches dependencies).
The process is dependant on your network status.

The key issues tensorflow pull its own distribution of protobuffers \(3.8.1\) and eigen \(3.3.9\). These key information is stored in bazel file "tensorflow/workspace.bzl".

If any mismatch happens, checkout the tensorflow bazel file to see if any dependancies out of date and use the scripts provided by use to install the proper one.

> bash scipts/build_tensorflow.sh

If script does not work \(due to your network proxy, git ssh configuration, git cache setting and many other reasons\), here are the procedures to do:

1\) open "vendors/github.com/tensorflow_cc/tensorflow_cc/cmake/TensorflowBase.cmake" and modify:

```
ExternalProject_Add(
  tensorflow_base

## Replace git repotory to released archived files 

  # GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  # GIT_TAG "${TENSORFLOW_TAG}"

  URL https://github.com/tensorflow/tensorflow/archive/${TENSORFLOW_TAG}.tar.gz  

  TMP_DIR "/tmp"
  STAMP_DIR "tensorflow-stamp"
  # DOWNLOAD_DIR "tensorflow"
  SOURCE_DIR "tensorflow"
  BUILD_IN_SOURCE 1
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/build_tensorflow.sh"
  INSTALL_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_links.sh" bazel-bin
)
```

where $TENSORFLOW_TAG defined in "tensorflow_cc/cmake/CMakefile.cmake"

2\) since bazel consumes a large portion of memories \[1\]\[2\] which could break your building process, replace bazel build command with the following one:

```
# tensorflow_cc/cmake/build_tensorflow.sh.in
sudo bazel build --jobs=8 --config=monolithic tensorflow:libtensorflow_all.so
```

## Data

All the data reside in "$ROOT/data/${DATASET\_SOURCE}/${DATASET\_NAME}". Curerntly I have tested TUM hand hold camera dataset with rgbd ground truths for references.
More and more dataset will be supported soon

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
[4] SVSO documentation and tutorial : https://drive.google.com/file/d/1XWf_esVG2gso-aZVtplwyTEWV8a8iSFa/view?usp=sharing, archived on 4th Jun, 2020
```
