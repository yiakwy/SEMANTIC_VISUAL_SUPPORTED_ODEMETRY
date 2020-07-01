#include <gtest/gtest.h>
#include <locale>
#include <iostream>
#include <fstream>
#include "env_config.h"

char* getFormattedTime(void);

char* getFormattedTime(void)
{
    time_t rawtime;
    struct tm* timeinfo;
    
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    
    // Must be static, otherwise won't work
    static char _repr_val[20];
    strftime(_repr_val, sizeof(_repr_val), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    return _repr_val;
    
}


class TestEnvironment: public ::testing::Environment {
public:
    
    static std::string getStartTime() {
        static const std::string timestamp( getFormattedTime() );
        return timestamp;
    }
    
    virtual void SetUp() { getStartTime(); }
    
};

static std::streambuf* pipebuf = nullptr;

int main(int argc, char** argv)
{
    try {
        ::testing::InitGoogleTest(&argc, argv);
        ::testing::AddGlobalTestEnvironment(new TestEnvironment);
        
        std::string mylog = LOG_DIR + "/gtest.out";
        std::ofstream out(mylog);
        
        pipebuf = std::cout.rdbuf();
        // std::cout.rdbuf(out.rdbuf());
        
        // ::testing::GTEST_FLAG(filter) = "AsyncPubsubTest*";
        return RUN_ALL_TESTS();
    } catch (std::exception &e) {
        std::cerr << "Unhandled Exception: " << e.what() << std::endl;
        return 1;
    }
}
