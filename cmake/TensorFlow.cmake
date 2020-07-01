# building tensorflow by building vendors/github.com/tensorflow_cc
find_package(TensorFlow REQUIRED)
include_directories(
  ${TensorFlow_INCLUDE_DIRS}
  )
