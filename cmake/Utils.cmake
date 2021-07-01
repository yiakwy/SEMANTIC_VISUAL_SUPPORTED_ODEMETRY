# Author Lei Wang 
# Date 2019.10

function(echo)
	foreach(e ${ARGN})
		message(STATUS ${e})
	endforeach()
endfunction()

function(create_test testName)
    set (testSuitName "test_${testName}")
    echo ("Creating test ${testSuiteName}")
    add_executable(${testSuiteName}
            EXCLUDE_FROM_ALL
            "testMain.cpp"
            "gtest/${testName}/*.cpp")

    target_link_libraries(${testSuiteName}
            ${GTEST_BOTH_LIBRARIES}
            ${ARGN}
            ${GLOG_LIBRARIES}
            )

    # this will go away once we get rid of catkin
    add_test(NAME testSuites
            COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testSuiteName}
            )
    add_dependencies(check ${testSuiteName})
endfunction()

# following syntax from bazel add_library
function(create_libary library_name srcs)
    echo ("Creating library ${library_name}...")
    add_library(${library_name} SHARED
            ${srcs})
    set_target_properties(${library_name} PROPERTIES LINKER_LANGUAGE CXX PRIVATE ${LIB_LINKER_FLAGS})
    target_include_directories(${library_name}
            PUBLIC
            ${EIGEN3_INCLUDE_DIRS}
            ${PROTOBUF_INCLUDE_DIRS}
            )
    target_link_libraries(${library_name}
            base
            )

    if (CMAKE_HOST_WIN32)
        INSTALL(TARGETS ${library_name} DESTINATION lib/win64/${library_name})
    elseif (CMAKE_HOST_APPLE)
        INSTALL(TARGETS ${library_name} DESTINATION lib/darwin/${library_name})
    elseif (CMAKE_HOST_UNIX)
        INSTALL(TARGETS ${library_name} DESTINATION lib/linux/${library_name})
    endif()
    INSTALL(DIRECTORY ./ DESTINATION include/${library_name} FILES_MATCHING PATTERN ".hpp .h")

    add_custom_target(install_${library_name}
            "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target install
            DEPENDS ${library_name}
            COMMENT "installing ${library_name} ..."
            )

endfunction()