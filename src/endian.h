#pragma once

#include "byteswap.h"

#include <bit>

template <typename T, std::endian E>
class Endian {
public:
  inline Endian(): m_data(convert(T{})) {}
  inline Endian(const T& other): m_data(convert(other)) {}
  inline operator T() const { return convert(m_data); }

private:
  T m_data;

  static T convert(const T &value) {
    if constexpr (std::endian::native == E) {
      return value;
    }
    else {
      return byteswap(value);
    }
  }
};

template <typename T>
using BE = Endian<T, std::endian::big>;

template <typename T>
using LE = Endian<T, std::endian::little>;
