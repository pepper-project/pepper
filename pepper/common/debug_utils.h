
#ifndef CODE_PEPPER_COMMON_DEBUG_UTILS_H_
#define CODE_PEPPER_COMMON_DEBUG_UTILS_H_

#include <cassert>
#include <iostream>

#define assert_error(msg) assert(!(std::cerr << msg << std::endl))

template<typename T> bool
inRange(const T& val, const T& min, const T& max)
{
  return (val >= min) && (val < max);
}

#endif

