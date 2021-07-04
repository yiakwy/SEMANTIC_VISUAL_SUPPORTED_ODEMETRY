set (PROTO_DIR   "${PROJECT_SOURCE_DIR}/proto")
set (PROTO_CODEC "${PROTO_GEN}")

file(MAKE_DIRECTORY ${PROTO_CODEC})

# Find Protobuf installation
set (protobuf_MODULE_COMPATABLE TRUE)
set (Protobuf_PROTOC_EXECUTABLE "/usr/local/bin/protoc") # defaults to protoc-3.11.1, now downgraded to protoc-3.8.0 to meet requirements of tensorflow

# set (Protobuf_PROTOC_EXECUTABLE "/usr/bin/protoc") # defaults to protoc-3.0.0
# include(FindProtobuf)
# check cmake --help-module FindProtobuf for details
find_package(Protobuf REQUIRED)
echo ("Using protobuf ${Protobuf_VERSION}")
echo ("Protobuf_INCLUDE_DIRS: ${Protobuf_INCLUDE_DIRS}")
include_directories(${Protobuf_INCLUDE_DIRS})

set(PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

# Later we will add GRPC as our network transportation layer if we need it
# Upon which, I have build pubsub as network application to enable IPC and 
# distributed communication.

# Find gRPC instalation
# list (APPEND CMAKE_PREFIX_PATH "/usr/local/include/grpc++" "/usr/local/include/google/protobuf/")
# find_package(GRPC REQUIRED)
# echo ("Using GRPC ${gRPC_VERSION}")

# set(GRPC_GRPCPP_UNSECURE "-L/usr/local/lib -lgrpc++ -lgrpc -lgrpc++_reflection -ldl")
# set(GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

# Proto file
file(GLOB_RECURSE PROTO_FILES "${PROTO_DIR}/*.proto")

foreach(HW_PROTO_FIL ${PROTO_FILES})
  get_filename_component(hw_proto_abs_tmp ${HW_PROTO_FIL}  ABSOLUTE)
  get_filename_component(hw_proto_path_tmp "${hw_proto}" PATH)
  list(FIND hw_proto ${hw_proto_abs_tmp} _contains_already)
  if (${_contains_already} EQUAL -1)
    list(APPEND hw_proto ${hw_proto_abs_tmp})
    list(APPEND hw_proto_path ${hw_proto_path_tmp})
  endif()
endforeach()

echo ("hw_proto : ${hw_proto}")
echo ("hw_proto_path : ${hw_proto_path}")

# utilities to generate proto c++ files, also see commandline tools, which
# will generate proto implementation for c++, python and golang simulatneously
function(PROTOBUF_GENERATE_CPP SRCS HDRS DEST)
  if (NOT ARGN)
    message(SEND_ERROR "PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  set (PROTO_FILES "${ARGN}")
  if (PROTOBUF_GENERATE_CPP_APPEND_PATH)
    foreach (FIL ${PROTO_FILES})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
        list (APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set (_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR}/proto)
  endif()

  if (DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
    set (Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
  endif()
  
  # might be set outside the script
  if (DEFINED Protobuf_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list (FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if (${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path)
      endif()
    endforeach()
  endif()

  # set variables
  set(${SRCS})
  set(${HDRS})

  echo ("PROTO_FILES: ${PROTO_FILES}")
  foreach(FIL ${PROTO_FILES})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    file(RELATIVE_PATH REL_FIL ${PROTO_DIR} ${FIL}) # keep relative path to proto_codec
    get_filename_component(REL_PATH "${REL_FIL}" PATH)
    get_filename_component(FIL_WE ${REL_FIL} NAME_WE)
    if (NOT REL_PATH STREQUAL "")
      set (FIL_WE ${REL_PATH}/${FIL_WE})
    endif()
    if (NOT PROTOBUF_GENERATE_CPP_APPEND_PATH)
      get_filename_component(FIL_DIR ${FIL} DIRECTORY)
      if (FIL_DIR)
        set (FIL_WE "${FIL_DIR}/${FIL_WE}")
      endif()
    endif()
    
    list(APPEND ${SRCS} "${DEST}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${DEST}/${FIL_WE}.pb.h")
  
    # call protoc upon changes of proto files
    add_custom_command(
      OUTPUT "${DEST}/${FIL_WE}.pb.cc"
             "${DEST}/${FIL_WE}.pb.h"
      COMMAND protobuf::protoc
      ARGS --cpp_out=${DEST} --proto_path=${PROTO_DIR} ${ABS_FIL}
      DEPENDS ${ABS_FIL} protobuf::protoc
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM
      )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set (${SRCS} ${${SRCS}} PARENT_SCOPE)
  set (${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

echo("Protobuf_PROTOC_EXECUTABLE: ${Protobuf_PROTOC_EXECUTABLE}")
if (Protobuf_PROTOC_EXECUTABLE)
  if (NOT TARGET protobuf::protoc)    
    add_executable(protobuf::protoc IMPORTED ../python/svso.cpp)
    if (EXISTS "${Protobuf_PROTOC_EXECUTABLE}")
      set_target_properties(protobuf::protoc PROPERTIES
        IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}")
    endif()
  endif()
endif()
