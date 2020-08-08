# building tensorflow by following instructions from github.com/cjweeks/tensorflow-cmake and use my provided cmake/Modules/FindTensorflow.cmake
# find_package(Tensorflow REQUIRED)

# building tensorflow by building vendors/github.com/tensorflow_cc
find_package(TensorflowCC REQUIRED)

include_directories(
  ${TensorFlow_INCLUDE_DIRS}
  )

