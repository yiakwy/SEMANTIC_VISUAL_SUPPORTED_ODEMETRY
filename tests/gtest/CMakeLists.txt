echo ("GTEST_LIBRARIES: ${GTEST_LIBRARIES}")
echo ("GTEST_BOTH_LIBRARIES": ${GTEST_BOTH_LIBRARIES})

# using our new micro to create tests
create_test(model
        base
        model
        # ${GeographicLib_LIBRARIES}
        # ${PCL_LIBRARIES}
        ${TBB_LIBRARIES}
        boost_system
        boost_filesystem
        ${GLOG_LIBRARIES}
        )

