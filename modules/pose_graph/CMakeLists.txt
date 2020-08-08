project(pose_graph C CXX)

file(GLOB_RECURSE FILES *.cpp *.h *.hpp)

add_library(${PROJECT_NAME} SHARED
        ${FILES} point3d.cpp point3d.h pixel2d.cpp pixel2d.h camera.cpp camera.h mapblock.cpp mapblock.h)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX PRIVATE ${LIB_LINKER_FLAGS})
target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${EIGEN3_INCLUDE_DIRS}
        ${PROTOBUF_INCLUDE_DIRS}
        )
target_link_libraries(${PROJECT_NAME}
        base
        )

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

# @todo TODO
# link cuda if cuda devices found