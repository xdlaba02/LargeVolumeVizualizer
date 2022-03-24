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

template <typename T, uint32_t DENOM>
inline T div(const Vc::SimdArray<T, simd::len> &nom) {
  return (nom * FAST_DIV_MULT<T, DENOM>) >> FAST_DIV_SHIFT<T, DENOM>;
}
