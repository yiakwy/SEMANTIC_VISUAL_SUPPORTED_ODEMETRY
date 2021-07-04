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
// new API
/*
 *
 */
#include <linux/aio_abi.h>
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
#include <functional>
#include <future>

#include <tbb/concurrent_queue.h>
// @todo TOOD(lock free queue)

#include <memory>
#include <boost/optional.hpp>

#include <gflags/gflags.h>

// logging utilities
#include <glog/logging.h>
#include "base/logging.h"

#include "base/io/sensors/img.h"

#include "base/parallel_tasking_sched/threading_pool.h"

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
class AIOCompletionQueue;

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
        aiocb_t *submitted_jobs[1] = {0};

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

        // make cpu do something else, the `while` loop usually is represented by a local poller

        // synchronize with kernel, may be implemented as a Wait function
        while(aio_error(&aiocb) == EINPROGRESS)
        {
            LOG(INFO) << "READING ...";
            // wait for SIGPOLL by sigtimedwait, here is an example from intel https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/libraries/intel-s-c-asynchronous-i-o-extensions-for-windows-operating-systems/intel-s-c-asynchronous-i-o-library-for-windows-operating-systems/example-for-aio-suspend-function.html
            submitted_jobs[0] = &aiocb;
            int ret = aio_suspend(submitted_jobs, 1, nullptr);
            if (ret == FAILURE) {
                LOG(INFO) << "aiocb is not ready ...";
            }
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

    size_t get_length(const int fd, const std::string& fn, int* ok) {
        size_t memory_size;
        struct stat attrib;
        // read info from linux innode metadata
        if (stat(fn.c_str(), &attrib) < 0) {
            LOG(ERROR)
                    << format("Could not find the meta data from the innode of the file[Linux] %s", fn.c_str());
            close(fd);
            goto __err_get_length_by_fd__;
        }
        memory_size = (size_t) attrib.st_size;
        return memory_size;
    __err_get_length_by_fd__:
        *ok = FAILURE;
        return 0;
    }

    size_t get_length(std::string fn, int* file_descriptor, int* ok)
    {
        size_t memory_size;

        int fd = open(fn.c_str(), O_RDONLY);
        if (fd == -1) {
            LOG(ERROR) << format("Failure to open %s", fn.c_str());
            close(fd);
            goto __err_get_length_by_fn__;
        }

        *file_descriptor = fd;

        memory_size = get_length(fd, fn, ok);
        if (*ok == FAILURE) {
            goto __err_get_length_by_fn__;
        }
        return memory_size;

    __err_get_length_by_fn__:
        // clean up
        *ok = FAILURE;
        exit(FAILURE);
    }

    void test_async_read_with_cq();

};

static std::string evt2str(const Event& ev);
class Event {
public:
    using Ptr = std::shared_ptr<Event>;
    using ConstPtr = std::shared_ptr<const Event>;
    using Type = Event;

    int fd = -1; // file descriptor associated to an event

    enum class EventType {
        EVENT_CHANNEL_TIMEOUT = -1,
        EVENT_CHANNEL_SHUTDOWN = 0,
        EVENT_KERNEL_COMPLETE = 1,
        EVENT_KERNEL_IN_PROC = 2,
        EVENT_KERNEL_NOT_READY = 3
    };

    enum class Status {
        EVENT_FAILURE = -1,
        EVENT_UNDEF = 0,
        EVENT_SUCCESS = 1
    };

    EventType type;
    Status status;
    // address to data
    void* tag;
    // data buffer
    size_t numOfbytes;
    char* buf = nullptr;
    size_t readed;

public:
    friend std::ostream& operator<< (std::ostream& out, const Event& evt) {
        out << evt2str(evt);
        return out;
    }

};

static std::string evt2str(const Event& ev)
{
    switch(ev.type) {
        case Event::EventType::EVENT_KERNEL_COMPLETE:
            return "complete";
        case Event::EventType::EVENT_CHANNEL_TIMEOUT:
            return "timeout";
        case Event::EventType::EVENT_CHANNEL_SHUTDOWN:
            return "shutdown";
        default:
            return "unkown";
    }
}

#define CQ_FAILURE false
#define CQ_STOP false
#define CQ_SUCCESS true

// See grpc C++ grpc::CompletionQueue (built on top of grpc C Core).
// Also see our IPC pubsub software package "grpc pubsub" which is residing in network/pubsub subfolder. For how to use it.
// We provide real examples to use it in a real complex network environment both for broker server and client to process
// requests and responses asynchronously.
class CompletionQueue {
public:
    CompletionQueue() : pool_(5) {}
    virtual ~CompletionQueue() { Clear(); };

    virtual void Clear() {};

    using ReturnType = std::future<bool>;

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
                    break;
                default:
                    LOG(ERROR) << "Not supported event type:" << " " << evt2str(ev) <<std::endl;
            }

        } // while

    }

    // register an event into the queue
    virtual bool Push(Event& ev) {}
    virtual ReturnType AsyncPush(Event& ev) {}

    // @todo TODO
    virtual void Shutdown() {}

    EventChannel& completion_channel() {
        return completion_channel_;
    }
    /*
    EventChannel& submission_channel() {
        return submission_channel_;
    }
     */
protected:
    // thread safe event completion queue
    EventChannel completion_channel_;
    svso::base::pts::ThreadPool pool_;
    /*
    // thread safe event submission queue
    EventChannel submission_channel_;
    */
    // aotmic variables to maintain submission and completed
    std::atomic<std::uint32_t> submission_;

    std::atomic<std::uint32_t> completed_;
};

class AIOUringCompletionQueue : public CompletionQueue {
public:
    // will be added soon ,,,
};

class AIOCompletionQueue : public CompletionQueue
{
public:
    AIOCompletionQueue() : CompletionQueue() {}
    virtual ~AIOCompletionQueue() {}

    using Base = CompletionQueue;
    using Type = AIOCompletionQueue;
    using Ptr = std::shared_ptr<Type>;

    using aiocb_t = struct aiocb;
    using aiocb_t_ptr = std::shared_ptr<aiocb_t>;

    inline static aiocb_t* allocate_aiocb(size_t size=1) {
        aiocb_t* addr = (aiocb_t*)malloc(size * sizeof(aiocb_t));
        memset(addr, 0, sizeof(aiocb_t));
        return addr;
    }

    inline static void deallocate_aiocb(aiocb_t* aiocb_p) {
        if (aiocb_p != nullptr) {
            free(aiocb_p);
        }
    }

    void Clear() override {}

    // register an event into the queue
    virtual bool Push(Event& evt) override {
        int fd  = evt.fd;

        aiocb_t_ptr aiocb_ptr(allocate_aiocb(), &AIOCompletionQueue::deallocate_aiocb);
        aiocb_t **submitted_jobs = {0};

        if (evt.buf != nullptr) {
            delete[] evt.buf;
        }
        evt.buf = new char[evt.numOfbytes];

        aiocb_ptr->aio_nbytes = evt.numOfbytes;
        aiocb_ptr->aio_fildes = fd;
        aiocb_ptr->aio_offset = 0;
        aiocb_ptr->aio_buf = evt.buf;

        evt.type = Event::EventType::EVENT_KERNEL_NOT_READY;
        if (aio_read(aiocb_ptr.get()) == -1) {
            LOG(ERROR)
                    << format("Could create requests to read I/O event asynchronously");
            close(fd);
            delete[] evt.buf;
            evt.buf = nullptr;

            return false;

        }
        evt.type = Event::EventType::EVENT_KERNEL_IN_PROC;
        submitted_jobs[0] = aiocb_ptr.get();

        submission_.fetch_add(1, std::memory_order_relaxed);
        auto added_to_cq_when_completed = [=, &evt]() {

            // synchronize with kernel, may be implemented as a Wait function
            while(aio_error(aiocb_ptr.get()) == EINPROGRESS)
            {
                LOG(INFO) << "READING ...";
                // wait for completion
                int ret = aio_suspend(submitted_jobs, 1, nullptr);
                if (ret == 1) {
                    LOG(ERROR) << "aiocb is not ready ...";
                }
            }

            int readed = aio_return(aiocb_ptr.get());

            if (readed != -1) {
                evt.readed = readed;
                if (readed != evt.numOfbytes) {
                    evt.status = Event::Status::EVENT_FAILURE;
                    LOG(INFO) << format("expected to read %d bytes, but read %d", evt.numOfbytes, readed);
                } else {
                    evt.status = Event::Status::EVENT_SUCCESS;
                    LOG(INFO) << format("read %d bytes", readed);
                }
            } else {
                evt.status = Event::Status::EVENT_FAILURE;
                evt.readed = 0;
                LOG(ERROR) << "Error!";
            }

            // added to completion queue
            completion_channel_.push(evt);
        };

        pool_.enqueue(added_to_cq_when_completed);

        return true;
    }

    // register an event into the queue
    virtual ReturnType AsyncPush(Event& evt) override {
        int fd  = evt.fd;

        aiocb_t_ptr aiocb_ptr(allocate_aiocb(), &AIOCompletionQueue::deallocate_aiocb);
        aiocb_t *submitted_jobs[1] = {0};

        if (evt.buf != nullptr) {
            delete[] evt.buf;
        }
        evt.buf = new char[evt.numOfbytes];

        aiocb_ptr->aio_nbytes = evt.numOfbytes;
        aiocb_ptr->aio_fildes = fd;
        aiocb_ptr->aio_offset = 0;
        aiocb_ptr->aio_buf = evt.buf;

        bool is_succ = true;
        evt.type = Event::EventType::EVENT_KERNEL_NOT_READY;
        if (aio_read(aiocb_ptr.get()) == -1) {
            is_succ = false;
        } else {
            LOG(INFO) << "READING ...";
            submission_.fetch_add(1, std::memory_order_relaxed);
        }
        evt.type = Event::EventType::EVENT_KERNEL_IN_PROC;
        submitted_jobs[0] = aiocb_ptr.get();

        auto wait_for_kernel = [=, &evt]() {
            if (!is_succ) {
                LOG(ERROR)
                        << format("Could create requests to read I/O event asynchronously");
                close(evt.fd);
                delete[] evt.buf;
                evt.buf = nullptr;
                return false;
            }

            // synchronize with kernel, may be implemented as a Wait function
            while(aio_error(aiocb_ptr.get()) == EINPROGRESS)
            {
                // wait for completion
                // it is also possible to use "aio_fsync" API
                int ret = aio_suspend(submitted_jobs, 1, nullptr);
                if (ret == FAILURE) {
                    LOG(ERROR) << "aiocb is not ready ...";
                }
            }

            int readed = aio_return(aiocb_ptr.get());
            LOG(INFO) << "COMPLETE READING.";
            evt.type = Event::EventType::EVENT_KERNEL_COMPLETE;

            if (readed != -1) {
                evt.readed = readed;
                if (readed != evt.numOfbytes) {
                    evt.status = Event::Status::EVENT_FAILURE;
                    LOG(INFO) << format("expected to read %d bytes, but read %d", evt.numOfbytes, readed);
                } else {
                    evt.status = Event::Status::EVENT_SUCCESS;
                    LOG(INFO) << format("read %d bytes", readed);
                }
            } else {
                evt.status = Event::Status::EVENT_FAILURE;
                evt.readed = 0;
                LOG(ERROR) << "Error!";
            }
            return true;
        };

        std::future<bool> fut = pool_.enqueue(wait_for_kernel);
        return fut;
    }

    // @todo TODO
    virtual void Shutdown() override {}

private:

};

void TestAIOReader::test_async_read_with_cq() {
    /*
     * Running example:
     *
     *   /home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/cmake-build-debug/bin/libaio_test
     *   I0430 23:09:31.163206 10575 libaio_test.cpp:491] READING ...
     *   I0430 23:09:31.163282 10575 libaio_test.cpp:576] let us do something else ...
     *   I0430 23:09:31.163478 10576 libaio_test.cpp:519] COMPLETE READING.
     *   I0430 23:09:31.163488 10576 libaio_test.cpp:529] read 506503 bytes
     *
     */

    AIOCompletionQueue::Ptr cq(new AIOCompletionQueue);

    int fd = -1;
    int ok = 1;
    size_t memory_size = get_length(FLAGS_image_name, &fd, &ok);

    Event evt;

    evt.fd = fd;
    evt.numOfbytes = memory_size;
    evt.type = Event::EventType::EVENT_KERNEL_NOT_READY;

    auto fut = cq->AsyncPush(evt);

    // do something else
    LOG(INFO) << "let us do something else ...";

    fut.wait();

    CHECK(evt.readed == evt.numOfbytes);

}

void Parse_args(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
}

int main(int argc, const char** argv)
{
    Parse_args(argc, (char**)argv);
    Init_GLog(argc, argv);

    TestAIOReader reader;
    // passed
    /*
    reader.open_and_read(FLAGS_image_name);
     */

    // passed
    reader.test_async_read_with_cq();

    return 0;

}