if (NOT __GFLAGS_INCLUDED)
 set (__GFLAGS_INCLUDED TRUE)

 find_package(GFlags)
 echo("GFLAGS_INCLUDE_DIRS: ${GFLAGS_INCLUDE_DIRS}")

 include_directories (
  ${GFLAGS_INCLUDE_DIRS} 
 )

 echo("GFLAGS_LIBRARIES: ${GFLAGS_LIBRARIES}")
 link_libraries(${GFLAGS_LIBRARIES})
endif()


