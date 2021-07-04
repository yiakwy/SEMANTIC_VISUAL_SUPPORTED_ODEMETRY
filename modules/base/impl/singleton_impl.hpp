#pragma once

#include "base/singleton.h"

namespace svso {
namespace base {

template <class T> pthread_once_t Singleton<T>::p_once_ = PTHREAD_ONCE_INIT;
template <class T> typename Singleton<T>::T_ptr Singleton<T>::instance_ = nullptr;

  } // base
} // svso