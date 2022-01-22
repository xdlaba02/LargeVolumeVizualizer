#pragma once

#include <type_traits>

#include <cstdint>

template<typename T, typename std::enable_if_t<sizeof(T) == 1, bool> = true>
inline T byteswap(T value) {
  return value;
}

template<typename T, typename std::enable_if_t<sizeof(T) == 2, bool> = true>
inline T byteswap(T value) {
  uint16_t &num = reinterpret_cast<uint16_t>(value);
  
  num = (num >> 8)
      | (num << 8);

  return value;
}

template<typename T, typename std::enable_if_t<sizeof(T) == 4, bool> = true>
inline T byteswap(T value) {
  uint32_t &num = reinterpret_cast<uint32_t>(value);

  num = (num >> 24)
      | ((num & 0x00ff0000) >> 8)
      | ((num & 0x0000ff00) << 8)
      | (num << 24);

  return value;
}

template<typename T, typename std::enable_if_t<sizeof(T) == 8, bool> = true>
inline T byteswap(T value) {
  uint64_t &num = reinterpret_cast<uint64_t>(value);

  num = (num >> 56)
      | ((num & 0x00ff000000000000) >> 40)
      | ((num & 0x0000ff0000000000) >> 24)
      | ((num & 0x000000ff00000000) >> 8)
      | ((num & 0x00000000ff000000) << 8)
      | ((num & 0x0000000000ff0000) << 24)
      | ((num & 0x000000000000ff00) << 40)
      | (num << 56);

  return value;
}
