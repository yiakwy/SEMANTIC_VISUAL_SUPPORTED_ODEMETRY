//
// Created by yiak on 2020/9/8.
//

// Linux Library
#include <sys/stat.h>
// see tutorial from https://fwheel.net/aio.html, also check these articles:
//   1. https://www.fsl.cs.sunysb.edu/~vass/linux-aio.txt
//   2. https://github.com/littledan/linux-aio ("littledan" illustrated how to submit I/O requests in parallel within
//      multi threads environment). Note instead of passing "aio" to linker, we pass "rt" to include the libaio library
//  projects highlights:
//   3. GRPC C Core : provide abstraction layer for I/O(mostly network), file loading, polling and concurrency management
//   4. libuv : core library for Node.js project, which should do a very good job in loading and pooling file descriptor cross platforms
//   5. libuv tutorial book: https://nikhilm.github.io/uvbook/index.html

// GNU software
#ifdef __APPLE__
#include <sys/types.h>
#include <errno.h>
#include <aio.h>
#endif

#ifdef __linux__
#include <aio.h>
#endif

#ifdef WIN32
// NOT IMPLEMENTED YET!
#endif

#define _POXIS_SOURCE
#include <unistd.h>
#include <fcntl.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <tbb/concurrent_queue.h>
// @todo TOOD(lock free queue)


#include <boost/optional.hpp>

#include <gflags/gflags.h>

// logging utilities
#include <glog/logging.h>
#include "base/logging.h"

#include "env_config.h"

using namespace svso::system;
using namespace svso::base::logging;

// Parsing comandline inputs
DEFINE_string(image_name,
          format("%s/tum/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png", env_config::DATA_DIR.c_str()),
          "image name");

#define BLOCK_SIZE 4*1024*1024
#define FAILURE -1

class Event;
// @todo TODO(lock free message queue)
using EventChannel = tbb::concurrent_bounded_queue<boost::optional<Event> >;

class CompletionQueue;

class AsyncReader {
public:
    virtual ~AsyncReader() {}

    virtual void open_and_read(const std::string& fn) = 0;
};

/*
 *
 * Reactors like POSIX apis select (Linux with O(n) scanning operation), epoll (Ubuntu Linux ...), kequeue (BSD Darwin)
 * use two different system calls to poll I/O operation and read data
 */
class TestAIOReader : public AsyncReader {
public:

    // test aio functionality
    void open_and_read(const std::string& fn) override {
        using aiocb_t = struct aiocb;

        char* buf;
        size_t buf_size;

        size_t memory_size;
        int readed;

        int fd = ::open(fn.c_str(), O_RDONLY, 0); // not sure whether AIO is best for O_DIRECT (i.e., access to devices without buffer)

        if (fd == -1) {
            LOG(ERROR) << format("Failure to open %s", fn.c_str());
            close(fd);
            goto __error__;
        }

        struct stat attrib;
        // read info from linux innode metadata
        if (stat(fn.c_str(), &attrib) < 0) {
            LOG(ERROR)
                    << format("Could not find the meta data from the innode of the file[Linux] %s", fn.c_str());
            close(fd);
            goto __error__;
        }
        memory_size = (size_t) attrib.st_size;

        aiocb_t aiocb;
        memset(&aiocb, 0, sizeof(aiocb));
        if (BLOCK_SIZE > memory_size) {
            buf = new char[BLOCK_SIZE];
            buf_size = BLOCK_SIZE;
        } else {
            buf = new char[memory_size];
            buf_size = memory_size;
        }

        aiocb.aio_nbytes = buf_size;
        aiocb.aio_fildes = fd;
        aiocb.aio_offset = 0;
        aiocb.aio_buf = buf;

        // enqueue requests to read
        if (aio_read(&aiocb) == -1) {
            LOG(ERROR)
                    << format("Could create requests to read %s asynchronously", fn.c_str());
            close(fd);
            delete[] buf;
            goto __error__;
        }

        // make cpu do something else, the wille loop usually represent a poller service
        while(aio_error(&aiocb) == EINPROGRESS)
        {
            LOG(INFO) << "READING ...";
        }

        readed = aio_return(&aiocb);

        if (readed != -1) {
            if (readed != memory_size) {
                LOG(INFO) << format("expected to read %d bytes, but read %d", memory_size, readed);
            } else {
                LOG(INFO) << format("read %d bytes", readed);
            }
        } else {
            LOG(ERROR) << "Error!";
        }

        close(fd);
        delete[] buf;
        return ;

        __error__:
        // clean up
        exit(FAILURE);
    }

};

class Event {
public:
    using Ptr = std::shared_ptr<Event>;
    using ConstPtr = std::shared_ptr<const Event>;
    using Type = Event;

    int fd = -1; // file descriptor associated to an event

    enum class EventType {
        EVENT_CHANNEL_TIMEOUT = -1,
        EVENT_CHANNEL_SHUTDOWN = 0,
        EVENT_KERNEL_COMPLETE = 1
    };

    enum class Status {
        EVENT_FAILURE = -1,
        EVENT_SUCCESS = 0,

    };

    EventType type;
    Status status;
    // address to data
    void* tag;
};

#define CQ_FAILURE false
#define CQ_STOP false
#define CQ_SUCCESS true

// See grpc C++ grpc::CompletionQueue (built on top of grpc C Core).
// Also see our IPC pubsub software package "grpc pubsub" which residing in network/pubsub subfolder for how to use it. We provide real examples to use it in a
// real complex network environment both for broker server and client to process requests and responses asynchronously.
class CompletionQueue {
public:
    CompletionQueue() {}
    virtual ~CompletionQueue() {}

    enum NextStatus {
        TIMEOUT = 0,
        SHUTDOWN = 1,
        GOT_EVENT = 2
    };

    // @todo TODO
    bool Next(void** tag, bool *ok) {
        while (true) {
            boost::optional<Event> datum;
            completion_channel_.pop(datum);
            if (!datum) {
                // got a signal to stop
                return CQ_STOP; // SHUTDOWN

            };

            auto &ev = datum.get();
            switch (ev.type) {
                case Event::EventType::EVENT_CHANNEL_TIMEOUT:
                case Event::EventType::EVENT_CHANNEL_SHUTDOWN:
                    return CQ_FAILURE;
                case Event::EventType::EVENT_KERNEL_COMPLETE:
                    *ok = ev.status != Event::Status::EVENT_FAILURE;
                    *tag = ev.tag;
                    // todo TODO(finalize the result and check status) :
                    if (true) {
                        return CQ_SUCCESS;
                    }
                    // see reference to transport https://grpc.github.io/grpc/core/md_doc_core_transport_explainer.html
                    //  1. stream ops for handshake
                    //  2. handel interruption from the environment
                    //  3. also see issue #16842: https://github.com/grpc/grpc/pull/16842
                    break;
            }
        }

    }

    // register an event into the queue
    void Push(Event ev) {
        
    }

    // @todo TODO
    void Shutdown() {}

    EventChannel& completion_channel() {
        return completion_channel_;
    }

    EventChannel& submission_channel() {
        return submission_channel_;
    }

private:
    // thread safe event completion queue
    EventChannel completion_channel_;

    // thread safe event submission queue
    EventChannel submission_channel_;

    // aotmic variables to maintain submission and completed
    std::atomic<std::uint32_t> submission_;

    std::atomic<std::uint32_t> completed_;
};


void Parse_args(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}

int main(int argc, const char** argv)
{
    Parse_args(argc, (char**)argv);
    Init_GLog(argc, argv);

    TestAIOReader reader;
    reader.open_and_read(FLAGS_image_name);

    return 0;

}