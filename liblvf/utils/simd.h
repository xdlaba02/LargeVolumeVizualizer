/**
* @file simd.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Vector types and helper operations for them.
*/

#pragma once

#include "fast_div.h"

#include <Vc/Vc>

namespace simd {
  constexpr uint32_t len = Vc::float_v::size();

  using float_v = Vc::float_v;
  using float_m = Vc::float_m;


  using uint32_v = Vc::SimdArray<uint32_t, len>;
  using  int32_v = Vc::SimdArray<int32_t, len>;
};

template <typename T, typename M>
inline void swap(T &a, T &b, const M &mask) {
  T c = a;
  a(mask) = b;
  b(mask) = c;
}

template <uint32_t N>
inline simd::uint32_v div(const simd::uint32_v &nom) {
  return (nom * FAST_DIV_MULT<uint32_t, N>) >> FAST_DIV_SHIFT<uint32_t, N>;
}

template <uint32_t N>
inline simd::int32_v div(const simd::int32_v &nom) {
  return (nom * FAST_DIV_MULT<int32_t, N>) >> FAST_DIV_SHIFT<int32_t, N>;
}
