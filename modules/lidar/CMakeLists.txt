project(lidar C CXX)

file(GLOB_RECURSE FILES *.cpp *.h *.hpp)

list(REMOVE_ITEM FILES "${CMAKE_CURRENT_SOURCE_DIR}/pc_preprocessor/pc_preprocessor_test.cpp")

create_libary(${PROJECT_NAME} "${FILES}")

target_link_libraries(${PROJECT_NAME}
        pose_graph
        )