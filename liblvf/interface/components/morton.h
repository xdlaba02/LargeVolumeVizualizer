#pragma once

#include <cstdint>

#include <bit>

template <uint32_t BITS>
class Morton {
public:
    template <typename T>
    static inline constexpr T interleave(T v) {
      return interleaver<std::bit_width(BITS - 1)>(v);
    }

    template <typename T>
    static inline constexpr T combine_interleaved(T x, T y, T z) {
      return x | (y << 1) | (z << 2);
    }

    template <typename T>
    static inline constexpr T to_index(T x, T y, T z) {
      return combine_interleaved(interleave(x), interleave(y), interleave(z));
    }

    template <typename T>
    static inline constexpr T deinterleave(T v) {
      return deinterleaver<std::bit_width(BITS - 1)>(v);
    }

    template <typename T>
    static inline constexpr void separate_interleaved(T v, T &x, T &y, T &z) {
      x = (v >> 0) & magic_number(1);
      y = (v >> 1) & magic_number(1);
      z = (v >> 2) & magic_number(1);
    }

    template <typename T>
    static inline constexpr void from_index(T v, T &x, T &y, T &z) {
      separate_interleaved(v, x, y, z);
      x = deinterleave(x);
      y = deinterleave(y);
      z = deinterleave(z);
    }

private:

  static inline consteval uint32_t magic_number(uint32_t bits) {
    uint32_t result = (1 << bits) - 1;
    for (uint32_t shift = bits * 3; shift < 32; shift <<= 1) {
      result |= result << shift;
    }
    return result;
  }

  template <uint32_t WIDTH, typename T, typename std::enable_if_t<WIDTH == 1, bool> = true>
  static inline constexpr T interleaver(T v) {
    return (v | (v << (1 << WIDTH))) & magic_number(1 << (WIDTH - 1));
  }

  template <uint32_t WIDTH, typename T, typename std::enable_if_t<WIDTH != 1, bool> = true>
  static inline constexpr T interleaver(T v) {
    v = (v | (v << (1 << WIDTH))) & magic_number(1 << (WIDTH - 1));
    return interleaver<WIDTH - 1, T>(v);
  }

  template <uint32_t WIDTH, typename T, typename std::enable_if_t<WIDTH == 1, bool> = true>
  static inline constexpr T deinterleaver(T v) {
    return (v | (v >> (1 << WIDTH))) & magic_number(1 << WIDTH);
  }

  template <uint32_t WIDTH, typename T, typename std::enable_if_t<WIDTH != 1, bool> = true>
  static inline constexpr T deinterleaver(T v) {
    v = deinterleaver<WIDTH - 1, T>(v);
    return (v | (v >> (1 << WIDTH))) & magic_number(1 << WIDTH);
  }
};
