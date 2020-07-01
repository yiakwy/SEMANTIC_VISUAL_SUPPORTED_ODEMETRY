include (FindPackageHandleStandardArgs)

set (GFLAGS_ROOT_DIR "" CACHE PATH "GFLAG_ROOT")

find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h)
find_library(GFLAGS_LIBRARY gflags)

find_package_handle_standard_args(GFlags DEFAULT_MSG GFLAGS_INCLUDE_DIR GFLAGS_LIBRARY)

if (GFLAGS_FOUND)
  set (GFLAGS_INCLUDE_DIRS "${GFLAGS_INCLUDE_DIR}/gflags")
  set (GFLAGS_LIBRARIES ${GFLAGS_LIBRARY})
  message(STATUS "Found gflags (include: ${GFLAGS_INCLUDE_DIR}, library: ${GFLAGS_LIBRARY})")
  mark_as_advanced(GFLAGS_LIBRARY GFLAGS_INCLUDE_DIR)
endif()
