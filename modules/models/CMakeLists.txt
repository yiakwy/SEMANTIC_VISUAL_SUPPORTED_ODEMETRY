project(models C CXX)

# freshly added
find_package(TensorflowCC REQUIRED)

#include and src file
file(GLOB_RECURSE FILES *.cpp *.h *.hpp *.c)

execute_process(COMMAND python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
echo("${PROJECT_NAME} PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
set (PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
execute_process(COMMAND python -c "import distutils.sysconfig as sysconfig; print('%s/%s' %
(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('INSTSONAME')))"
        OUTPUT_VARIABLE PYTHON_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
echo("${PROJECT_NAME} PYTHON_LIBRARY: ${PYTHON_LIBRARY}")

# anaconda doesn't ship symlink to 'python-config'
execute_process(COMMAND python3.6-config --exec-prefix
        OUTPUT_VARIABLE PYLIBPATH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND echo ${PYLIBPATH_PREFIX}/lib
        OUTPUT_VARIABLE PYLIBPATH OUTPUT_STRIP_TRAILING_WHITESPACE)

set(Python_ADDITIONAL_VERSIONS 3.6)
find_package(PythonLibs 3.6)

list(REMOVE_ITEM FILES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
list(REMOVE_ITEM FILES "${CMAKE_CURRENT_SOURCE_DIR}/simple_mrcnn_infer.cpp")
echo ("FILES : ${FILES}")
#shared library
add_library(${PROJECT_NAME} SHARED
        ${FILES}
        # ${ROOT}/vendors/matplotlib-cpp/matplotlibcpp.h
		)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX PRIVATE ${LIB_LINKER_FLAGS})
# include third party libraries here

# add python library support
target_include_directories(${PROJECT_NAME} PUBLIC ${PYTHON_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${PYTHON_LIBRARIES}
        base
	    TensorflowCC::TensorflowCC # freshly added
        )

add_definitions( -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} )
add_definitions( -DPYTHON_LIBRARY=${PYTHON_LIBRARY} )
add_definitions( -DPYTHONHOME="${PYLIBPATH_PREFIX}")

set(ENV{PYTHONHOME} "${PYLIBPATH_PREFIX}")
echo("PYTHONHOME: $ENV{PYTHONHOME}")

if (CMAKE_HOST_WIN32)
  INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib/win64/${PROJECT_NAME})
elseif (CMAKE_HOST_APPLE)
  INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib/darwin/${PROJECT_NAME})
elseif (CMAKE_HOST_UNIX)
  INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib/linux/${PROJECT_NAME})
endif ()
INSTALL(DIRECTORY ./ DESTINATION include/${PROJECT_NAME} FILES_MATCHING PATTERN ".hpp")

# main.cpp
add_custom_target(install_${PROJECT_NAME}
        "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target install
        DEPENDS ${PROJECT_NAME}
        COMMENT "Installing ${PROJECT_NAME} ...")

add_executable(${PROJECT_NAME}_cli
        ${TensorFlow_INCLUDE_DIRS}
        main.cpp
        )

# add links
set_target_properties(${PROJECT_NAME}_cli PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${PROJECT_NAME}_cli PUBLIC ${TensorFlow_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}_cli
        PUBLIC
        ${PROJECT_NAME}
        TensorflowCC::TensorflowCC
	# ${TensorFlow_LIBRARIES} # build by following instructions from github.com/cjweeks/tensorflow-cmake and use my provided cmake/Modules/FindTensorflow.cmake
        )

# @todo TODO
# link cuda if cuda devices found

# simple_mrcnn_infer.cpp
add_executable(simple_mrcnn_infer
        ${TensorFlow_INCLUDE_DIRS}
        simple_mrcnn_infer.cpp
        )

# add links
set_target_properties(simple_mrcnn_infer PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(simple_mrcnn_infer PUBLIC ${TensorFlow_INCLUDE_DIRS})

target_link_libraries(simple_mrcnn_infer
        PUBLIC
        ${PROJECT_NAME}
        TensorflowCC::TensorflowCC)

# @todo TODO
# link cuda if cuda devices found