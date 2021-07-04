//
// Created by yiak on 2021/7/4.
//

#include <set>
#include <vector>

#include <gflags/gflags.h>

#include <tbb/concurrent_hash_map.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "base/logging.h"

#include "base/array_like_hasher.h"

using namespace svso::base::logging;

void test_tbb_concurrent_map_hasher() {
    using PointKey = size_t;
    using RangeKey = std::pair<int, int>;
    /* used for inverse query table */
    using RangeBucket = tbb::concurrent_hash_map<RangeKey, std::set<PointKey>, svso::base::_pair_hasher<RangeKey>>;

    RangeBucket range_bucket;
    LOG(INFO) << format("[%s] has passed", "test_tbb_concurrent_map_hasher");
}

void test_array_alike_hasher() {
    std::vector<int> test_array = {0, 1, 2};
    size_t ret_of_test_array = svso::base::_hasher().template operator()<std::vector<int>>(test_array);

    Eigen::VectorXi test_eigen_vector;
    test_eigen_vector.resize(3);
    test_eigen_vector[0] = 0;
    test_eigen_vector[1] = 1;
    test_eigen_vector[2] = 2;
    size_t ret_of_test_eigen_vector = svso::base::_hasher().template operator()<Eigen::VectorXi>(test_eigen_vector);

    std::pair<int, int> test_pair;
    test_pair.first = 0;
    test_pair.second = 1;
    size_t ret_of_test_pair = svso::base::_hasher().template operator()<std::pair<int, int>>(test_pair);

    LOG(INFO) << format("[%s] has passed", "test_array_alike_hasher");

}

void Parse_args(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}

/*
 * Run cmd (./bin/array_alike_hasher)
 *
 * [100%] Linking CXX executable ../../bin/array_alike_hasher_test
 * [100%] Built target array_alike_hasher_test
 * (py36) ➜  build-test git:(dev/wangyi/feature_add_cpp_impl) ✗ ./bin/array_alike_hasher_test
 * I0704 18:23:01.815645 23461 array_alike_hasher_test.cpp:29] [test_tbb_concurrent_map_hasher] has passed
 * I0704 18:23:01.815696 23461 array_alike_hasher_test.cpp:48] [test_array_alike_hasher] has passed
 *
 */
int main(int argc, const char** argv) {
    Parse_args(argc, (char**)argv);
    Init_GLog(argc, argv);

    test_tbb_concurrent_map_hasher();
    test_array_alike_hasher();
}
