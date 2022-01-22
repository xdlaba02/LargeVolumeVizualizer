#pragma once

namespace morton {
  template <typename T>
  inline constexpr T interleave_4b_3d(T v) {
    v = (v | (v << 4)) & 0x0C3;
    return (v | (v << 2)) & 0x249;
  }

  template <typename T>
  inline constexpr T combine_interleaved(T x, T y, T z) {
    return x | (y << 1) | (z << 2);
  }

  template <typename T>
  inline constexpr T to_index_4b_3d(T x, T y, T z) {
    return morton_combine_interleaved(interleave_4b_3d(x), interleave_4b_3d(y), interleave_4b_3d(z));
  }

  template <typename T>
  inline constexpr T deinterleave_4b_3d(T v) {
    v &= 0x249;
    v = (v | (v >> 2)) & 0x0C3;
    return (v | (v >> 4)) & 0x00F;
  }

  template <typename T>
  inline constexpr void from_index_4b_3d(T v, T &x, T &y, T &z) {
    x = deinterleave_4b_3d(v >> 0);
    y = deinterleave_4b_3d(v >> 1);
    z = deinterleave_4b_3d(v >> 2);
  }
}
