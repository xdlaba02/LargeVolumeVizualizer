/**
* @file fast_div.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief This one is a bit weird, but it is used to perform approximative integer division by a non-power-of-two constant via mults and shifts.
* I found out that although my compiler does this with normal 32 bit integers automatically, it does not do it with vector types.
* Those functions search for the best combination of the mult and shift values for specific denominator and type at compile time.
* Compilation is slow as heck, but it is much faster than the generic integer division when determining block that sample falls into while sampling blocked volume with duplicated edges.
* The precision is bad but it should be more than enough for our volume sizes.
*/

#pragma once

#include <cstdint>
#include <cmath>

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
