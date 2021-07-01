//
// Created by yiak on 2021/4/23.
//
#pragma once

#ifndef SEMANTIC_RELOCALIZATION_MISC_H
#define SEMANTIC_RELOCALIZATION_MISC_H

#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <limits>

#include <base/logging.h>

namespace svso {
namespace base {

class AtomicCounter {
public:
    using Ptr = std::shared_ptr<AtomicCounter>;
    using ConstPtr = std::shared_ptr<const AtomicCounter>;

    AtomicCounter() : counter_(0) {}
    virtual ~AtomicCounter() {}

    AtomicCounter(const AtomicCounter &other) {
        CopyFrom(other);
    }

    AtomicCounter(AtomicCounter &&other) noexcept
            : AtomicCounter() {
        *this = ::std::move(other);
    }

    inline AtomicCounter &operator=(const AtomicCounter &other)
    {
        CopyFrom(other);
        return *this;
    }

    inline AtomicCounter &operator=(AtomicCounter &&other) noexcept
    {
        if (this != &other) {
            CopyFrom(other);
        }
        return *this;
    }

    void CopyFrom(const AtomicCounter &other)
    {
        if (this == &other) return;
        Clear();

        // assignments
        counter_ = other.counter();
    }

    void Clear() {}

    const size_t counter() const { return counter_; }

    const size_t incr() {
        counter_.fetch_add(1, std::memory_order_relaxed);
        if (counter_ > std::numeric_limits<size_t>::max())
        {
            // @todo TODO use an array to represent the big number
            LOG(FATAL) << "The counter reaches its maximum in this platform. Please update the software to support arbitrary large number!";
        }
        return counter_;
    }

    // overriding operatgors

    // Unary operator:
    const size_t operator() () {
        return incr();
    }

private:
    std::atomic<size_t> counter_;
};

class Identity {
public:
    using Ptr = std::shared_ptr<Identity>;
    using Const = std::shared_ptr<const Identity>;

    static AtomicCounter Sequence;

    explicit Identity(uint64_t seq, std::string name="", uint64_t id=-1, std::string uuid="", std::string tok="") :
            seq_(seq),
            name_(name),
            id_(id),
            uuid_(uuid),
            tok_(tok)
    {}
    explicit Identity(std::string name="", uint64_t id=-1, std::string uuid="", std::string tok="") : Identity(Sequence.incr(), name, id, uuid, tok)
    {}

    virtual ~Identity() {}

    /*
     * attributes
     */

    Identity(const Identity &other) {
        CopyFrom(other);
    }

    Identity(Identity &&other) noexcept
            : Identity(0) {
        *this = ::std::move(other);
    }

    inline Identity &operator=(const Identity &other)
    {
        CopyFrom(other);
        return *this;
    }

    inline Identity &operator=(Identity &&other) noexcept
    {
        if (this != &other) {
            CopyFrom(other);
        }
        return *this;
    }

    void CopyFrom(const Identity &other)
    {
        if (this == &other) return;
        Clear();

        // assignments
        seq_ = other.seq();
        name_ = other.name();
        uuid_ = other.uuid();
        tok_ = other.tok();
    }

    void Clear() {}

    /*
     * protected/private attributes accessor
     */
    const uint64_t &seq() const { return seq_; }

    const uint64_t &set_seq(const uint64_t &seq) {
        seq_ = seq;
        return seq_;
    }

    const std::string &name() const { return name_; }

    const std::string &set_name(const std::string name) {
        name_ = name;
        return name_;
    }

    const uint64_t &id() const { return id_; }

    const uint64_t &set_id(const uint64_t &id) {
        id_ = id;
        return id_;
    }

    const std::string &uuid() const { return uuid_; }

    const std::string &set_uuid(const std::string uuid) {
        uuid_ = uuid;
        return uuid_;
    }

    const std::string &tok() const { return tok_; }

    const std::string &set_tok(const std::string tok) {
        tok_ = tok;
        return tok_;
    }

private:
    uint64_t seq_;
    std::string name_;
    int64_t id_;
    std::string uuid_;
    std::string tok_;

};

} // base
} // svso

#endif //SEMANTIC_RELOCALIZATION_MISC_H
