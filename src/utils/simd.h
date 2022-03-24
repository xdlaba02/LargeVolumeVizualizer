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

template <uint32_t DENOM, typename T>
inline T div(const T &nom) {
  using VTYPE = typename T::value_type;
  return (nom * FAST_DIV_MULT<VTYPE, DENOM>) >> FAST_DIV_SHIFT<VTYPE, DENOM>;
}
