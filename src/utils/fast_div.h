#pragma once

#include <cstdint>
#include <cmath>

#include <type_traits>

// fast division by 7 in 32 bit register  - maximum viable number this can divide correctly is 57343.
// fast division by 15 in 32 bit register - maximum viable number this can divide correctly is 74908.
// fast division by 31 in 32 bit register - maximum viable number this can divide correctly is 63487.
// fast division by 63 in 32 bit register - maximum viable number this can divide correctly is 64511.

template <typename T, T DENOM>
static consteval T fast_div_mult(const T &shift) {
  return std::ceil((1 << shift) / static_cast<double>(DENOM));
}

template <typename T, T DENOM>
static consteval T fast_div_shift() {
  T best_shift = 0;
  uint64_t best_i = 0;

  T shift = 0;
  uint64_t i = 0;
  while (i <= std::numeric_limits<T>::max() && shift < (sizeof(T) * 8) - 1) {
    T val = T(T(i) * fast_div_mult<T, DENOM>(shift)) >> shift;

    if (val != (i / DENOM)) {
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

template <typename T, T DENOM>
static const constinit T FAST_DIV_SHIFT = fast_div_shift<T, DENOM>();

template <typename T, T DENOM>
static const constinit T FAST_DIV_MULT = fast_div_mult<T, DENOM>(FAST_DIV_SHIFT<T, DENOM>);