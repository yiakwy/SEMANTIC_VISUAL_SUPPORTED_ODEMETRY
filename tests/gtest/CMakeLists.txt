echo ("GTEST_LIBRARIES: ${GTEST_LIBRARIES}")
echo ("GTEST_BOTH_LIBRARIES: ${GTEST_BOTH_LIBRARIES}")

# using our new micro to create tests
create_test(model
        base
        )

