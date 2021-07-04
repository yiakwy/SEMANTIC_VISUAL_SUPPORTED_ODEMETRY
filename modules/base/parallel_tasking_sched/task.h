//
// Author : Lei Wang (yiak.wy@gmail.com)
// Date : 2019
//

#pragma once

#ifndef MAPPING_TASK_H
#define MAPPING_TASK_H

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <thread>

// lock free multi threads tasking engine
#include <tbb/tbb.h>

// exception types for this application
#include "base/exceptions.h"

using namespace svso::base;

namespace svso {
namespace base {
// Parallel Tasking Scheduler (PTS)
namespace pts {

class ThreadPool;

// Used to cast from data address to task instance memory block
class TaskBase {
public:
    using Type = TaskBase;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    using tid_type = std::thread::id;

    virtual ~TaskBase() {}

    virtual void Reset() = 0;

    static tid_type self_thread() {
        return std::this_thread::get_id();
    }

};

template <typename F, typename... Args>
class Task : public TaskBase {
public:
    using Base = TaskBase;
    using Type = Task<F, Args...>;
    using Ptr = std::shared_ptr<Type>;
    using ConstPtr = std::shared_ptr<const Type>;

    using ReturnType = typename std::result_of<F(Args...)>::type;
    using FutureType = std::future<ReturnType>;

    Task() {}

    explicit Task(F& f) {
        packaged_task_with_args_ = std::make_shared<std::packaged_task<ReturnType(Args...)>>(
                std::forward<F>(f)
        );

    }

    explicit Task(F&& f) {
        packaged_task_with_args_ = std::make_shared<std::packaged_task<ReturnType(Args...)>>(
                std::forward<F>(f)
        );
    }

    virtual ~Task() {}

    // @todo TODO Remove return value, instead return future
    void operator()(Args &... args) {
        (*packaged_task_with_args_)(args...);
    }

    auto get_future() -> FutureType { // std::future<typename Type::ReturnType>
        return packaged_task_with_args_->get_future();
    }

    virtual void Reset() override {
        packaged_task_with_args_->reset();
    }

    void from(F& f, Args&... args) {
        packaged_task_with_args_ = std::make_shared<std::packaged_task<ReturnType()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    }

    void from(F&& f, Args&&... args) {
        packaged_task_with_args_ = std::make_shared<std::packaged_task<ReturnType()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    }

    static void* tag(Type* task) {
        return reinterpret_cast<void*>(task);
    }

protected:
    // the input task parameters will be captured by lambda to reduce the coding complexities
    std::shared_ptr<std::packaged_task<ReturnType(Args...)>> packaged_task_with_args_;
};

// recover task instance from void*
// see Usage in lidar/benchmark.h where we store different function as callbacks
static TaskBase* detag(void* address) {
    TaskBase* task = (reinterpret_cast<TaskBase*>(address));
    return task;
}

class TaskWrapper {
public:
    using Type = TaskWrapper;
    using Ptr = std::shared_ptr<Type>;

    TaskWrapper() {}

    template <typename F, typename... Args>
    void CreateTask(F& f) {
        task_ins_.reset(new Task<F, Args...>(f));
    }

    template <typename F, typename... Args>
    void CreateTask(F&& f) {
        task_ins_.reset(new Task<F, Args...>(f));
    }

    template<typename F, typename...Args>
    void CreateTaskWithArgs(F& f, Args&... args) {
        typename Task<F, Args...>::Ptr task(new Task<F, Args...>());
        task->from(f, args...);
        task_ins_.reset(task);
    }

    // @todo Remove return value, instread return future
    template <typename F, typename... Args>
    void operator()(Args &... args) {
        using TaskType = Task<F, Args...>;

        TaskType* task = dynamic_cast<TaskType*>(task_ins_.get());
        if (task == nullptr) {
            LOG(FATAL) << "Function type of F is not correct!";
        }
        (*task)(args...);
    }

    template <typename F, typename... Args>
    auto get_future() -> std::future<typename std::result_of<F(Args...)>::type> {
        using TaskType = Task<F, Args...>;
        using FutureType = typename Task<F, Args...>::FutureType;

        TaskType* task = dynamic_cast<TaskType*>(task_ins_.get());
        FutureType fut = task->get_future();
        return fut;
    }

    void Reset() {
        task_ins_->Reset();
    }

private:
    std::shared_ptr<TaskBase> task_ins_;
};

// extension of intel tbb task group implementation, see issue : https://github.com/oneapi-src/oneTBB/issues/180
class TaskPool {
 public:
  virtual ~TaskPool() {}
};

class TBBTaskPoolImpl : public TaskPool {
 public:
  using type = TBBTaskPoolImpl;
  using base_type = TaskPool;
  using tbb_task_group_t = std::shared_ptr<tbb::task_group>;

  TBBTaskPoolImpl() {}

  virtual ~TBBTaskPoolImpl() {}

  template <typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using ReturnType = typename std::result_of<F(Args...)>::type;
    using FutureType = std::future<ReturnType>;
    using FuncType = std::function<void(void)>;

    TaskWrapper::Ptr task(new TaskWrapper);
    task->template CreateTaskWithArgs<F, Args...>(f, args...);

    // push the task into queue


    FutureType fut = task->get_future<F, Args...>();
    task_group_->run([&] {
        (*task).template operator()<FuncType>();

    });
    return fut;
  }

  void wait() { task_group_->wait(); }

 protected:
  tbb_task_group_t task_group_;
  tbb::concurrent_queue<TaskWrapper::Ptr> tasks_;
};

    }
  }
}

#endif  // MAPPING_TASK_H
