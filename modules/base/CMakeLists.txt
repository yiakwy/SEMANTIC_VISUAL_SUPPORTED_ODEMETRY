project(base C CXX)

##### Proto #####
PROTOBUF_GENERATE_CPP(hw_proto_srcs hw_proto_hdrs ${PROTO_CODEC} ${hw_proto})

file(GLOB_RECURSE FILES *.cpp *.h *.hpp)
echo ("hw_proto_srcs: ${hw_proto_srcs}")
echo ("hw_proto_hdrs: ${hw_proto_hdrs}")
echo ("PROTOBUF_INCLUDE_DIRS: ${PROTOBUF_INCLUDE_DIRS}")
echo ("PROTOBUF_LIBRARIES: ${PROTOBUF_LIBRARIES}")

list(REMOVE_ITEM FILES "${CMAKE_CURRENT_SOURCE_DIR}/io/experimental/libaio_test.cpp")
list(REMOVE_ITEM FILES "${CMAKE_CURRENT_SOURCE_DIR}/array_alike_hasher_test.cpp")
list(REMOVE_ITEM FILES "${CMAKE_CURRENT_SOURCE_DIR}/io/config_manager_test.cpp")
##### Base #####

# for base library, instead of using `create_tool`, we keep maximum flexibility to modify configuration

include_directories(${PROTO_CODEC})
add_library(${PROJECT_NAME} SHARED
        ${FILES}
        ${hw_proto_srcs}
        ${hw_proto_hdrs})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX PRIVATE ${LIB_LINKER_FLAGS})
target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${EIGEN3_INCLUDE_DIRS}
        ${PROTOBUF_INCLUDE_DIRS}
        ${TBB_INCLUDE_DIRS}
        )
target_link_libraries(${PROJECT_NAME}
        boost_system boost_filesystem
        opencv_core opencv_imgproc opencv_imgcodecs
        ${TBB_LIBRARIES}
        ${PROTOBUF_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${PCL_LIBRARIES}
        )

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  target_link_libraries(${PROJECT_NAME}
    opencv_highgui)
endif()

if (CMAKE_HOST_WIN32)
  INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib/win64/${PROJECT_NAME})
elseif (CMAKE_HOST_APPLE)
  INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib/darwin/${PROJECT_NAME}) 
elseif (CMAKE_HOST_UNIX)
  INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib/linux/${PROJECT_NAME})
endif()
INSTALL(DIRECTORY ./ DESTINATION include/${PROJECT_NAME} FILES_MATCHING PATTERN ".hpp .h")

add_custom_target(install_${PROJECT_NAME}
  "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target install
  DEPENDS ${PROJECT_NAME}
  COMMENT "installing ${PROJECT_NAME} ..."
  )

##### Local Test

### libaio test
add_executable(libaio_test
        io/experimental/libaio_test.cpp
        )

# add links
set_target_properties(libaio_test PROPERTIES LINKER_LANGUAGE CXX PRIVATE "${LIB_LINKER_FLAGS} -lrt")
target_link_libraries(libaio_test
        PUBLIC
        ${PROJECT_NAME}
        rt
        )

### array alike hasher test
add_executable(array_alike_hasher_test
        array_alike_hasher_test.cpp)

# add links
set_target_properties(array_alike_hasher_test PROPERTIES LINKER_LANGUAGE CXX PRIVATE "${LIB_LINKER_FLAGS}")
target_link_libraries(array_alike_hasher_test
        PUBLIC
        ${PROJECT_NAME}
        )

### config manager test
add_executable(config_manager_test
        io/config_manager_test.cpp)

# add links
set_target_properties(config_manager_test PROPERTIES LINKER_LANGUAGE CXX PRIVATE "${LIB_LINKER_FLAGS}")
target_link_libraries(config_manager_test
        PUBLIC
        ${PROJECT_NAME})
