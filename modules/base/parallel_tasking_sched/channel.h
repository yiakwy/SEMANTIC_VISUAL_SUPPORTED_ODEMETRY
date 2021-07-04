//
//  channel.hpp
//  pubsub
//
//  Created by Lei Wang 王磊 on 2019/2/15.
//  REF:
//       https://thispointer.com/c11-multithreading-part-7-condition-variables-explained/
// The key ideas of this implementation is to allow threads to use read and
// write locks respectively. The `recv` method also allows timeout so that the
// channel could collect enough information.

#pragma once

#ifndef channel_hpp
#define channel_hpp

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>

/* Defined in c++14 and in c++11
using namespace std::chrono_literals;
*/

#include <iostream>
#include <sstream>

#include <stdexcept>

// #include "clientregistry.hpp"
namespace svso {
namespace base {
// Parallel Tasking Scheduler
namespace pts {
// GRPC pubsub
namespace netowrk {
namespace pubsub {

struct Owner;

template <class Message>
class Channel {
 public:
  Channel() : state_(OPEN){}

  Channel(std::string name, Owner *owner)
      : name_(name),
        owner_(owner),
        state_(OPEN),
        readed_(0),
        wrote_(0),
        r_waiting_(0) {
    if (owner == nullptr) {
      std::stringstream stream;
      stream << "[network::pubsub::Channel::Constructor] [ERROR] should not "
                "pass nullptr as owner."
             << std::endl;
      std::cout << stream.str();
      // Only for c++11
      const char *msg = stream.str().c_str();
      stream.str("");
      throw std::runtime_error(msg);
    }
  }

  // @todo : TODO
  Channel(Channel &ch) {}

  // @todo : TODO
  Channel(Channel &&ch) {}

  ~Channel() {
    queue.clear();
    owner_ = nullptr;
  }

  void close() {
    std::unique_lock<std::mutex> lock(mutex_);
    state_ = CLOSED;
    condVar.notify_all();
  }

  bool is_closed() {
    if (state_ == CLOSED) {
      return true;
    } else {
      return false;
    }
  }

  bool recv(Message &out, bool wait = true) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (wait) {
      r_waiting_++;
      condVar.wait(lock, [&]() { return is_closed() || !queue.empty(); });
      r_waiting_--;
    }
    if (queue.empty()) {
      return false;
    }
    out = queue.front();
    queue.pop_front();
    return true;
  }

  // supporting timeout
  template <class Rep, class Period>
  bool recv(Message &out, bool wait = true,
            const std::chrono::duration<Rep, Period> &rel_time =
                std::chrono::seconds(1) /* 1s */) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();
    if (wait) {
      r_waiting_++;
      if (condVar.wait_for(lock, rel_time, [&]() {
            return is_closed() || !queue.empty();
          }) == false) {
        r_waiting_--;
        // timeout, not done
        std::stringstream stream;
        stream << "[network::pubsub::Channel::recv] [ERROR] "
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now() - start)
                      .count()
               << " milli secs timeout, no messages published into channel <"
               << name_ << ">." << std::endl;
        std::cout << stream.str();
        stream.str("");
        return false;
      } else {  // log wait time
        r_waiting_--;
        if (!queue.empty()) {
          std::stringstream stream;
          stream << "[network::pubsub::Channel::recv] [INFO] "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now() - start)
                        .count()
                 << " milli secs timeout, " << queue.size()
                 << " messages left into channel <" << name_ << ">"
                 << ", subscribed/published = " << wrote_ << "/" << readed_
                 << std::endl;
          std::cout << stream.str();
          stream.str("");
        }
      }
    }

    if (is_closed()) {
      std::stringstream stream;
      stream << "[network::pubsub::Channel::recv] [INFO] channel <" << name_
             << "> is closed." << std::endl;
      std::cout << stream.str();
      stream.str("");
      return false;
    }
    if (!queue.empty()) {
      out = queue.front();
      queue.pop_front();
      wrote_++;
      std::stringstream stream;
      stream << "[network::pubsub::Channel::recv] [INFO] recv a message from "
                "channel <"
             << name_ << ">. " << queue.size() << " left"
             << ", subscribed/published = " << wrote_ << "/" << readed_
             << std::endl;
      std::cout << stream.str();

      if (name_ == "") {
        std::stringstream stream;
        stream << "[network::pubsub::Channel::recv] [ERROR] recv channel "
                  "associated to an undefined client."
               << std::endl;
        std::cout << stream.str();
        stream.str("");
      }

      return true;
    } else {
      std::stringstream stream;
      stream << "[network::pubsub::Channel::recv] [WARNNING] the queue "
                "associated to the channel is empty. aka no messages published "
                "to channel <"
             << name_ << ">." << std::endl;
      std::cout << stream.str();
      stream.str("");
      return false;
    }
  }

  void send(const Message &in) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t origin_size = queue.size();
    if (is_closed()) {
      throw std::logic_error("put to a closed channel.");
    }
    queue.push_back(in);
    readed_++;
    std::stringstream stream;
    stream << "[network::pubsub::Channel::send] [INFO] Message <" << in.data()
           << "> published into the channel <" << name_ << ">, " << origin_size
           << " messages increased to " << queue.size()
           << ", subscribed/published = " << wrote_ << "/" << readed_
           << std::endl;
    std::cout << stream.str();

    if (name_ == "") {
      std::stringstream stream;
      stream << "[network::pubsub::Channel::send] [ERROR] The channel "
                "associated to an undefined client."
             << std::endl;
      std::cout << stream.str();
      stream.str("");
    }

    if (r_waiting_ > 0) {
      condVar.notify_one();
    }
  }

  void set_name(std::string name) { name_ = name; }

  std::string get_name() { return name_; }

 private:
  enum STATE { CLOSED, OPEN };

  std::list<Message> queue;
  std::mutex mutex_;
  // std::mutex mu_r_;
  // std::mutex mu_w_; // for write operations purpose for queue with writing
  // buffer size
  std::condition_variable condVar;

  // std::condition_variable cond_r;
  // std::condition_variable cond_w;
  std::string name_;
  // static_cast<Your Owner Type>(owner_) -> ins
  // void* owner_;
  Owner *owner_;
  std::atomic<std::uint32_t> readed_;
  std::atomic<std::uint32_t> wrote_;
  // uint32_t w_waiting = 0;
  std::atomic<std::uint32_t> r_waiting_;
  STATE state_;
};

}  // pubsub
}  // network
}  // pts
}  // base
}  // svso
#endif /* channel_hpp */
