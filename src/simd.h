#pragma once

#include "fast_div.h"

#include <Vc/Vc>

namespace simd {
  using namespace Vc;

  constexpr uint32_t len = float_v::size();

  using uint32_v = SimdArray<uint32_t, len>;
  using  int32_v = SimdArray<int32_t, len>;

  template <typename T, typename M>
  inline void swap(T &a, T &b, const M &mask) {
    T c = a;
    a(mask) = b;
    b(mask) = c;
  }

  template <uint32_t D, typename T>
  inline T fast_div(const T &n) {
    return (n * FastDiv<typename T::value_type, D>::MULT) >> FastDiv<typename T::value_type, D>::SHIFT;
  }
};
