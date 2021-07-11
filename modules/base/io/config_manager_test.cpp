//
// Created by yiak on 2021/7/11.
//
#include <gflags/gflags.h>

#include <glog/logging.h>
#include "base/logging.h"

#include "base/io/config_manager.h"

DEFINE_string(yaml_conf, "config_root.yml", "yaml test file");

using namespace svso::base::logging;
using namespace svso::base::io;

void test_attributes() {
    Attributes attr;

    std::string key_prefix = "test_key";

    // *** test raw data
    int test_int = 1;
    size_t test_unsigned_long_integer = 2;
    float test_float = 3.1;
    double test_double = 3.2;
    const std::string test_string = "this is merely for test purpose!";

    // *** test artifical composite data
    Number test_number_1, test_number_2, test_number_3, test_number_4;
    test_number_1 = test_int;
    test_number_2 = test_unsigned_long_integer;
    test_number_3 = test_float;
    test_number_4 = test_double;

    String test_string_obj;
    test_string_obj = test_string;

    // *** test writing : insert into attrs
    ObjectPtr object_of_number(new Object);
    *object_of_number = test_number_1;
    std::string key_of_test_number_1 = key_prefix + "_test_number_1";
    attr.insert(key_of_test_number_1, object_of_number);
    LOG(INFO) << format("writing <%s, %d> to attr", key_of_test_number_1.c_str(), test_int);

    object_of_number.reset(new Object);
    *object_of_number = test_number_2;
    std::string key_of_test_number_2 = key_prefix + "_test_number_2";
    attr.insert(key_of_test_number_2, object_of_number);
    LOG(INFO) << format("writing <%s, %zu> to attr", key_of_test_number_2.c_str(), test_unsigned_long_integer);

    object_of_number.reset(new Object);
    *object_of_number = test_number_3;
    std::string key_of_test_number_3 = key_prefix + "_test_number_3";
    attr.insert(key_of_test_number_3, object_of_number);
    LOG(INFO) << format("writing <%s, %f> to attr", key_of_test_number_3.c_str(), test_float);

    object_of_number.reset(new Object);
    *object_of_number = test_number_4;
    std::string key_of_test_number_4 = key_prefix + "_test_number_4";
    attr.insert(key_of_test_number_4, object_of_number);
    LOG(INFO) << format("writing <%s, %f> to attr", key_of_test_number_4.c_str(), test_double);

    ObjectPtr object_of_string(new Object);
    *object_of_string = test_string_obj;
    std::string key_of_test_string_obj = key_prefix + "_test_string_obj";
    attr.insert(key_of_test_string_obj, object_of_string);
    LOG(INFO) << format("writing <%s, \"%s\"> to attr", key_of_test_string_obj.c_str(), test_string.c_str());

    // *** test reading
    ObjectPtr object_1 = attr.get(key_of_test_number_1);
    Value test_val_1 = boost::get<Value>(*object_1);
    Number test_number_from_val_1 = boost::get<Number>(test_val_1);
    int test_int_from_val_1 = boost::get<int>(test_number_from_val_1.val);
    CHECK(test_int_from_val_1 == test_int);
    LOG(INFO) << format("reading <%s, %d> from attr", key_of_test_number_1.c_str(), test_int_from_val_1);

    ObjectPtr object_2 = attr.get(key_of_test_number_2);
    Value test_val_2 = boost::get<Value>(*object_2);
    Number test_number_from_val_2 = boost::get<Number>(test_val_2);
    size_t test_unsigned_long_integer_from_val_2 = boost::get<size_t>(test_number_from_val_2.val);
    CHECK(test_unsigned_long_integer_from_val_2 == test_unsigned_long_integer);
    LOG(INFO) << format("reading <%s, %zu> from attr", key_of_test_number_2.c_str(), test_unsigned_long_integer_from_val_2);

    ObjectPtr object_3 = attr.get(key_of_test_number_3);
    Value test_val_3 = boost::get<Value>(*object_3);
    Number test_number_from_val_3 = boost::get<Number>(test_val_3);
    float test_float_from_val_3 = boost::get<float>(test_number_from_val_3.val);
    CHECK_NEAR(test_float_from_val_3, test_float, 1e-6);
    LOG(INFO) << format("reading <%s, %f> from attr", key_of_test_number_3.c_str(), test_float_from_val_3);

    ObjectPtr object_4 = attr.get(key_of_test_number_4);
    Value test_val_4 = boost::get<Value>(*object_4);
    Number test_number_from_val_4 = boost::get<Number>(test_val_4);
    double test_double_from_val_4 = boost::get<double>(test_number_from_val_4.val);
    CHECK_NEAR(test_double_from_val_4, test_double, 1e-6);
    LOG(INFO) << format("reading <%s, %f> from attr", key_of_test_number_4.c_str(), test_double_from_val_4);

    ObjectPtr object_5 = attr.get(key_of_test_string_obj);
    Value test_val_5 = boost::get<Value>(*object_5);
    String test_string_obj_from_val_5 = boost::get<String>(test_val_5);
    std::string test_string_from_val_5 = test_string_obj_from_val_5.val;
    CHECK(test_string_from_val_5.compare(test_string) == 0);
    LOG(INFO) << format("reading <%s, \"%s\"> from attr", key_of_test_string_obj.c_str(), test_string_from_val_5.c_str());

    LOG(INFO) << format("[%s] has passed", "test_attributes");
}

void test_config_manager() {
    ConfigManager manager;

    // this will be done automatically internally
    std::string conf_path = manager.get_absolute_path(FLAGS_yaml_conf);

    LOG(INFO) << format("Reading %s (%s)", FLAGS_yaml_conf.c_str(), conf_path.c_str());
    manager.load(conf_path);

    int confs;
    manager.get<int>("numberOfConf", &confs);
    std::string calibration_def;
    manager.get<std::string>("calibration/device", &calibration_def);

    manager.load_module_conf_path();
    std::string pointcloud_preprocessor_conf;
    manager.get<std::string>("lidar/pc_preprocessor", &pointcloud_preprocessor_conf);

    LOG(INFO) << format("Reading %d conf", confs);
    LOG(INFO) << format("Reading calibration device : %s", calibration_def.c_str());
    LOG(INFO) << format("Reading module PointCloud preprocessor configuration file : %s", pointcloud_preprocessor_conf.c_str());

    LOG(INFO) << format("[%s] has passed", "test_config_manager");
}

void Parse_args(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}


/*
 * Run cmd (./bin/config_manager_test):
 *
 * (py36) ➜  build-test git:(dev/wangyi/feature_add_cpp_impl) ✗ ./bin/config_manager_test
 * I0712 01:14:03.063218 22906 config_manager_test.cpp:43] writing <test_key_test_number_1, 1> to attr
 * I0712 01:14:03.063266 22906 config_manager_test.cpp:49] writing <test_key_test_number_2, 2> to attr
 * I0712 01:14:03.063274 22906 config_manager_test.cpp:55] writing <test_key_test_number_3, 3.100000> to attr
 * I0712 01:14:03.063282 22906 config_manager_test.cpp:61] writing <test_key_test_number_4, 3.200000> to attr
 * I0712 01:14:03.063288 22906 config_manager_test.cpp:67] writing <test_key_test_string_obj, this is merely for test purpose!> to attr
 * I0712 01:14:03.063294 22906 config_manager_test.cpp:75] reading <test_key_test_number_1, 1> from attr
 * I0712 01:14:03.063298 22906 config_manager_test.cpp:82] reading <test_key_test_number_2, 2> from attr
 * I0712 01:14:03.063303 22906 config_manager_test.cpp:89] reading <test_key_test_number_3, 3.100000> from attr
 * I0712 01:14:03.063308 22906 config_manager_test.cpp:96] reading <test_key_test_number_4, 3.200000> from attr
 * I0712 01:14:03.063311 22906 config_manager_test.cpp:103] reading <test_key_test_string_obj, this is merely for test purpose!> from attr
 * I0712 01:14:03.063315 22906 config_manager_test.cpp:105] [test_attributes] has passed
 * I0712 01:14:03.063360 22906 config_manager_test.cpp:114] Reading config_root.yml (/home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/config_root.yml)
 * I0712 01:14:03.064615 22906 config_manager.h:337] Setting calibration/device to path </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/test_data/20210703/test_device.yaml>
 * I0712 01:14:03.064632 22906 config_manager.h:337] Setting modules to path </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/modules.yaml>
 * I0712 01:14:03.065112 22906 config_manager.h:337] Setting msgs_node/cyber to path </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/msgs_node/cyber/bag_interface_sim_driver_node.conf>
 * I0712 01:14:03.065125 22906 config_manager.h:337] Setting msgs_node/ros to path </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/msgs_node/ros/bag_interface_sim_driver_node.conf>
 * I0712 01:14:03.065131 22906 config_manager.h:337] Setting lidar/pc_preprocessor to path </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/lidar/pc_preprocessor/pointcloud_preprocessor.conf>
 * I0712 01:14:03.065138 22906 config_manager.h:337] Setting lidar/detectors/object to path </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/lidar/detectors/object/object.conf>
 * I0712 01:14:03.065145 22906 config_manager.h:337] Setting lidar/detectors/object to path </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/lidar/detectors/object/pointpillar.conf>
 * I0712 01:14:03.065150 22906 config_manager.h:337] Setting lidar/detectors/object to path </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/lidar/detectors/object/pointrcnn.conf>
 * I0712 01:14:03.065156 22906 config_manager.h:337] Setting lidar/detectors/ground to path </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/lidar/detectors/ground/ground.conf>
 * I0712 01:14:03.065171 22906 config_manager_test.cpp:126] Reading 1 conf
 * I0712 01:14:03.065176 22906 config_manager_test.cpp:127] Reading calibration device : /home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/test_data/20210703/test_device.yaml
 * I0712 01:14:03.065179 22906 config_manager_test.cpp:128] Reading module PointCloud preprocessor configuration file : /home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/config/lidar/pc_preprocessor/pointcloud_preprocessor.conf
 * I0712 01:14:03.065182 22906 config_manager_test.cpp:130] [test_config_manager] has passed
 *
 */
int main(int argc, const char** argv)
{
    Parse_args(argc, (char**)argv);
    Init_GLog(argc, argv);

    test_attributes();
    test_config_manager();

    return 0;
}