project(pose_graph C CXX)

file(GLOB_RECURSE FILES *.cpp *.h *.hpp)

create_libary(${PROJECT_NAME} "${FILES}")