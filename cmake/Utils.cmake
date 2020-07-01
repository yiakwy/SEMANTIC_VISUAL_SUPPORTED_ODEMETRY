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
