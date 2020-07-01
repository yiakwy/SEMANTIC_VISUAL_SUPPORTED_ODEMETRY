project(base C CXX)

PROTOBUF_GENERATE_CPP(hw_proto_srcs hw_proto_hdrs ${PROTO_CODEC} ${hw_proto})

file(GLOB_RECURSE FILES *.cpp *.h *.hpp)
echo ("hw_proto_srcs: ${hw_proto_srcs}")
echo ("hw_proto_hdrs: ${hw_proto_hdrs}")
echo ("PROTOBUF_INCLUDE_DIRS: ${PROTOBUF_INCLUDE_DIRS}")
echo ("PROTOBUF_LIBRARIES: ${PROTOBUF_LIBRARIES}")
add_library(base SHARED 
        ${FILES}
        ${hw_proto_srcs}
        ${hw_proto_hdrs})
set_target_properties(base PROPERTIES LINKER_LANGUAGE CXX PRIVATE ${LIB_LINKER_FLAGS})
target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${EIGEN3_INCLUDE_DIRS}
        ${PROTOBUF_INCLUDE_DIRS}
        )
target_link_libraries(${PROJECT_NAME}
        boost_system
        boost_filesystem
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        ${PROTOBUF_LIBRARIES}
        ${GLOG_LIBRARIES}
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
INSTALL(DIRECTORY ./ DESTINATION include/${PROJECT_NAME} FILES_MATCHING PATTERN ".hpp")

add_custom_target(install_${PROJECT_NAME}
  "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target install
  DEPENDS ${PROJECT_NAME}
  COMMENT "installing ${PROJECT_NAME} ..."
  )

