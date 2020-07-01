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

// Warning : I havn't implemented copy and assignment constructor for
// r-reference (rhs) values.
// I left job intentionally for the future sprint.

class ThreadPool;

template <typename ReturnType>
class TaskBase {
 public:
  using base_type = void;
  using type = TaskBase;

  using Ptr = std::shared_ptr<type>;
  using ConstPtr = std::shared_ptr<const type>;

  using tid_type = std::thread::id;

  explicit TaskBase() {}

  virtual void operator()() = 0;
  /*
  virtual ReturnType get_future() = 0;
   */
  virtual auto get_future() -> std::future<ReturnType> = 0;

  //
  static tid_type self_thread() { return std::this_thread::get_id(); }

  virtual ~TaskBase() {}
};

template <typename F, typename... Args>
class Task : public TaskBase<typename std::result_of<F(Args...)>::type> {
 public:
  using ReturnType = typename std::result_of<F(Args...)>::type;
  using base_type = TaskBase<ReturnType>;
  using type = Task<F, Args...>;

  using Ptr = std::shared_ptr<type>;
  using ConstPtr = std::shared_ptr<const type>;

  using FutureType = std::future<ReturnType>;
  using CallType = std::function<ReturnType(void)>;

  explicit Task(F&& f, Args&&... args) {
    packaged_task_ = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
  }

  explicit Task(F& f, Args&... args) {
    packaged_task_ = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
  }

  virtual ~Task() {}

  virtual void operator()() override {
    auto func = [&, this]() { (*packaged_task_)(); };
    func();
  }

  /*
  FutureType get_future() override {
      return packaged_task_->get_future();
  }
   */
  auto get_future() -> std::future<typename std::result_of<F(Args...)>::type> {
    return packaged_task_->get_future();
  }

 protected:
  std::shared_ptr<std::packaged_task<ReturnType()>> packaged_task_;
};

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
  void* tag(typename Task<F, Args...>::Ptr task) {
    void* tag = reinterpret_cast<void*>(&task);
    return tag;
  }

  template <typename F, typename... Args>
  TaskBase<typename std::result_of<F(Args...)>::type>* detag(void* tag) {
    using ReturnType = typename std::result_of<F(Args...)>::type;
    TaskBase<ReturnType>* task =
        (*reinterpret_cast<typename Task<F, Args...>::Ptr*>(tag)).get();
    return task;
  }

  template <typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using ReturnType = typename std::result_of<F(Args...)>::type;

    typename Task<F, Args...>::Ptr task(new Task<F, Args...>(f, args...));

    // push the task into queue
    void* tag = tag(task);
    tasks_.push_back(tag);
    typename Task<F, Args...>::FutureType fut = task->get_future();
    task_group_->run([&] { (*task)(); });
    return fut;
  }

  void wait() { task_group_->wait(); }

 protected:
  tbb_task_group_t task_group_;
  std::vector<void*> tasks_;
};

    }
  }
}

#endif  // MAPPING_TASK_H
