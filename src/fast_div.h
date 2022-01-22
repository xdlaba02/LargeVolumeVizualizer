#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

#include <type_traits>

template <typename T, T SHIFT, T D>
constexpr T fast_div_mult() {
  return std::ceil((1 << SHIFT) / double(D));
}

template <typename T, T D>
constexpr T fast_div_shift() {
  T best_shift = 0;
  uint64_t best_i = 0;

  T shift = 0;
  uint64_t i = 0;
  while (i <= std::numeric_limits<T>::max() && shift < (sizeof(T) * 8) - 1) {
    T val = T(T(i) * T(std::ceil((1 << shift) / double(D)))) >> shift;

    if (val != (i / D)) {
      if (i > best_i) {
        best_shift = shift;
        best_i = i;
      }

      shift++;
    }
    else {
      i++;
    }
  }


  return best_shift;
}

#if 1

// fast division by 7 in 32 bit register  - maximum viable number this can divide correctly is 57343.
// fast division by 15 in 32 bit register - maximum viable number this can divide correctly is 74908.
// fast division by 31 in 32 bit register - maximum viable number this can divide correctly is 63487.
// fast division by 63 in 32 bit register - maximum viable number this can divide correctly is 64511.

// fast div for vector types
template <uint32_t D, typename T, std::enable_if_t<std::is_integral<typename T::value_type>::value, bool> = true>
inline T fast_div(T n) {
  constexpr typename T::value_type SHIFT = fast_div_shift<typename T::value_type, D>();
  constexpr typename T::value_type MULT = fast_div_mult<typename T::value_type, SHIFT, D>();
  return (n * MULT) >> SHIFT;
}

// fast div for integral types
template <uint32_t D, typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline T fast_div(T n) {
  constexpr T SHIFT = fast_div_shift<T, D>();
  constexpr T MULT = fast_div_mult<T, SHIFT, D>();
  return (n * MULT) >> SHIFT;
}

#else

template <uint32_t D, typename T>
inline T fast_div(T n) {
  return n / D;
}

#endif
