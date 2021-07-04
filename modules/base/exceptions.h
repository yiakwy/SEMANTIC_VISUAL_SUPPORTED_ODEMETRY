//
// Created by LEI WANG.
//

#pragma once

#ifndef MAPPING_EXCEPTIONS_H
#define MAPPING_EXCEPTIONS_H

#include <exception>

namespace svso {
namespace base {
namespace exceptions {

class NotImplemented : public std::exception {
  const char* what() const throw() {
    return "Bad Access Exception: not implemented yet!";
  }
};

/**
 *
 * Usage Exmple :
 * virtual func () {
 *   NOT_IMPLEMENTED
 * }
 */
#define NOT_IMPLEMENTED     \
  do {                      \
    throw NotImplemented(); \
  } while (0);

class WrongValue : public std::exception {
    const char* what() const throw () {
        return "Wrong value";
    }
};

    } // exceptions
  } // base
} // svso

#endif  // MAPPING_EXCEPTIONS_H
