include ("cmake/External/GFlags.cmake")

if (NOT __GLOG_INCLUDED)
  set (__GLOG_INCLUDED TRUE)

  find_package(Glog)
  echo("GLOG_INCLUDE_DIRS: ${GLOG_INCLUDE_DIRS}")

  include_directories(
   ${GLOG_INCLUDE_DIRS}
  )

  echo("GLOG_LIBRARIES: ${GLOG_LIBRARIES}")
  link_libraries(${GLOG_LIBRARIES})
endif()
