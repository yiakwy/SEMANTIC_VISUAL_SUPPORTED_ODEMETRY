//
// Created by yiak on 2021/4/23.
//
#pragma once

#ifndef SEMANTIC_RELOCALIZATION_TIMMER_H
#define SEMANTIC_RELOCALIZATION_TIMMER_H

#include <sys/time.h>
#include <chrono>
// std::put is not supported by g++ < 7.5
#include <iomanip>
#include <ctime>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

namespace svso {
namespace base {
namespace timmer {

using Clock = std::chrono::system_clock;

inline static Clock::time_point Now() { return Clock::now(); }

inline static std::string ISO8601UTCTime() {
    auto now = Now();
    auto iut = Clock::to_time_t(now);

    std::ostringstream os;
    os << std::put_time(std::gmtime(&iut), "%FT%TZ");

    // modified to meet requirements from gcc-4.8
    /*
    char buf[24];
    strftime(buf, sizeof(buf), "%FT%TZ", std::gmtime(&iut));
    os << buf;
     */
    return os.str();
}

// timer helper class to replace stupid NVIDIA StopWatchLinux which inherits from StopWatchInterface
// a nicer timer implementation for Linux machine

class TicToc {
public:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

    /**
     * Constructor, will start timing
     */
    TicToc() {
        tic();
        cur_ = start_;
    };

    /**
     * Reset the timer
     */
    void tic();

    /**
     * Auto adaptive timer, output elapsed time in adapted time units
     * @param message_prefix
     * @param tic_after_toc
     * @return example: "time elapsed: 80 ms;"
     */
    std::string toc(const std::string &message_prefix = "",
                    bool tic_after_toc = true);

    /**
     * Compute duration period last call
     */

    template<class DurationType, class Period>
    typename std::chrono::duration<DurationType, Period> elapse() {
        return cur_ - start_;
    }

    /* c++14
    auto elapse() {
      return cur_ - start_;
    }
     */

    int64_t toc_nano(bool tic_after_toc = true);

    int64_t toc_micro(bool tic_after_toc = true);

    int64_t toc_milli(bool tic_after_toc = true);

    int64_t toc_sec(bool tic_after_toc = true);

    int64_t toc_min(bool tic_after_toc = true);

private:
    TimePoint start_;
    TimePoint cur_;

};

inline void TicToc::tic() {
    start_ = std::chrono::high_resolution_clock::now();
}

inline std::string TicToc::toc(const std::string &message_prefix,
                               bool tic_after_toc) {
    using namespace std::chrono;

    cur_ = high_resolution_clock::now();
    auto diff = cur_ - start_;

    // auto conversion of print units
    std::stringstream ss;

    int64_t nano = duration_cast<nanoseconds>(diff).count();
    int64_t micro = duration_cast<microseconds>(diff).count();
    int64_t milli = duration_cast<milliseconds>(diff).count();
    int64_t sec = duration_cast<seconds>(diff).count();
    int64_t min = duration_cast<minutes>(diff).count();

    if (nano < 1000) {
        ss << message_prefix << nano << " ns ";
    } else if (micro < 1000) {
        ss << message_prefix << micro << " µs ";
    } else if (milli < 1000) {
        ss << message_prefix << milli << " ms ";
    } else if (sec < 600) {
        ss << message_prefix << sec << " s ";
    } else {
        ss << message_prefix << min << " min ";
    }

    if (tic_after_toc) this->tic();
    return ss.str();
}

inline int64_t TicToc::toc_nano(bool tic_after_toc) {
    cur_ = std::chrono::high_resolution_clock::now();
    auto nano = std::chrono::duration_cast<std::chrono::nanoseconds>(
            cur_ - start_).count();
    if (tic_after_toc) this->tic();
    return nano;
}

inline int64_t TicToc::toc_micro(bool tic_after_toc) {
    cur_ = std::chrono::high_resolution_clock::now();
    auto micro = std::chrono::duration_cast<std::chrono::microseconds>(
            cur_ - start_).count();
    if (tic_after_toc) this->tic();
    return micro;
}

inline int64_t TicToc::toc_milli(bool tic_after_toc) {
    cur_ = std::chrono::high_resolution_clock::now();
    auto milli = std::chrono::duration_cast<std::chrono::milliseconds>(
            cur_ - start_).count();
    if (tic_after_toc) this->tic();
    return milli;
}

inline int64_t TicToc::toc_sec(bool tic_after_toc) {
    cur_ = std::chrono::high_resolution_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(
            cur_ - start_).count();
    if (tic_after_toc) this->tic();
    return sec;
}

inline int64_t TicToc::toc_min(bool tic_after_toc) {
    cur_ = std::chrono::high_resolution_clock::now();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(
            cur_ - start_).count();
    if (tic_after_toc) this->tic();
    return min;
}

} // timmer
} // base
} // svso

#endif //SEMANTIC_RELOCALIZATION_TIMMER_H
