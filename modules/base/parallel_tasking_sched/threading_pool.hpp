/**
 * Author: Lei Wang, (yiak.wy@gmail.com)
 * Date: Created on Mar 26, 2019
 *       Updated on Sep 8, 2019
 * Reference: LLVM 3.8 ThreadingPool.hpp/cpp
 */

#pragma once

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace svso {
namespace base {
// Parallel Tasking Scheduler (PTS)
namespace pts {

/**
 * A naive implementation of threads pool, handy tools for simple task, not
 * fully integrated with intel tbb (under work, see implementation of task.hpp and tasking_pool.hpp)
 */
class ThreadPool {
 public:
  using Ptr = std::shared_ptr<ThreadPool>;
  using ConstPtr = std::shared_ptr<ThreadPool>;
  using Mutex = std::mutex;

  explicit ThreadPool(int);

  void Init(size_t threads) {
    // lock scope
    {
      std::unique_lock<std::mutex> lock(taskingMutex_);
      status_ = OPEN;
    }
    for (size_t i = 0; i < threads; i++) {
      workers_.emplace_back([this] {
        auto logging_info = [=](bool add, unsigned int active, size_t tasks) {
          std::stringstream stream;
          if (add) {
            stream << "[PTS::ThreadPool::Init] [INFO] add a task, live/tasks = "
                   << std::to_string(active) << "/" << std::to_string(tasks)
                   << std::endl;
          } else {
            stream
                << "[PTS::ThreadPool::Init] [INFO] remove a task, live/tasks = "
                << std::to_string(active) << "/" << std::to_string(tasks)
                << std::endl;
          }
          std::cout << stream.str();
        };
        while (true) {
          std::function<void()> task;
          // lock scope
          {
            std::unique_lock<std::mutex> lock(taskingMutex_);
            taskingCondVar_.wait(
                lock, [&] { return is_closed() || !tasks_.empty(); });

            if (is_closed() || tasks_.empty()) return;
            // lock scope, wait for completion of a running task
            {
              std::unique_lock<std::mutex> lock(completionMutex_);
              ++activeThreads_;
              // logging_info(true, activeThreads_.load(), tasks_.size());
            }
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
          {
            std::unique_lock<std::mutex> lock(completionMutex_);
            --activeThreads_;
            // logging_info(false, activeThreads_.load(), tasks_.size());
          }

          // notfiy task completion
          completionCondVar_.notify_all();
        }
      });
    }
  }

  void close() {
    std::unique_lock<std::mutex> lock(taskingMutex_);
    status_ = CLOSED;
    taskingCondVar_.notify_all();
  }

  bool is_closed() { return (status_ == CLOSED) ? true : false; }

  void wait() {
    std::unique_lock<std::mutex> lock(completionMutex_);
    auto logging_info = [=](unsigned int active, size_t tasks) {
      std::stringstream stream;
      stream << "[PTS::ThreadPool::wait] [INFO] live/tasks = "
             << std::to_string(active) << "/" << std::to_string(tasks)
             << std::endl;
      std::cout << stream.str();
    };
    logging_info(activeThreads_.load(), tasks_.size());
    completionCondVar_.wait(
        lock, [&] { return tasks_.empty() && !activeThreads_.load(); });
    logging_info(activeThreads_.load(), tasks_.size());
  }

  /**
   *
   * @tparam F
   * @tparam Args
   * @param f
   * @param args
   * @return std::future<Task::Return_Type>
   */
  template <class F, class... Args>
  auto enqueue(F &&f, Args &&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;

  std::size_t max_jobs() { return workers_.size(); }

  ~ThreadPool();

 private:
  enum STATE { CLOSED, OPEN };

  // need to keep track of threads so we can join them
  std::vector<std::thread> workers_;
  // the task queue
  std::queue<std::function<void()>> tasks_;

  // synchronization
  Mutex taskingMutex_;
  std::condition_variable taskingCondVar_;

  Mutex completionMutex_;
  std::condition_variable completionCondVar_;

  std::atomic<unsigned> activeThreads_;

  STATE status_;
};

/**
 *
 * @param threads : size_t
 */
inline ThreadPool::ThreadPool(int threads) : status_(OPEN) {
  if (threads < 1) {
    threads = std::thread::hardware_concurrency() - 1;
  }

  Init(threads);
}

/**
 *
 * @tparam F
 * @tparam Args
 * @param f
 * @param args
 * @return
 */
template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using ReturnType = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<ReturnType()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<ReturnType> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(taskingMutex_);

    // don't allow enqueueing a closed threads pool
    if (is_closed())
      throw std::runtime_error(
          "[PTS::ThreadPool::enqueue] [ERROR] Enqueue a task on a closed "
          "ThreadPool instance, aka no resources allocated to process the "
          "task.");

    tasks_.emplace([task]() { (*task)(); });
  }
  taskingCondVar_.notify_one();
  return res;
}

/**
 * Release hold resources.
 */
inline ThreadPool::~ThreadPool() {
  // lock scope
  {
    std::unique_lock<std::mutex> lock(taskingMutex_);
    status_ = CLOSED;
  }
  taskingCondVar_.notify_all();
  for (std::thread &worker : workers_) worker.join();
}

        }
    }
}

#endif
