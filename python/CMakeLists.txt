include_directories(${ROOT})

# pybind11 (version 2.2.1)
list (APPEND CMAKE_MODULE_PATH ${ROOT}/cmake/pybind11/tools)
include_directories(${ROOT}/cmake/pybind11/include)
include(pybind11Tools)

# add pysvso: Python Implementation of svso
add_subdirectory(pysvso)

