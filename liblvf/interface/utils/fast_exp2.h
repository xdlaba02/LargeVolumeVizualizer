/**
* @file fast_exp2.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief This performs 2^i with signed integer exponent just by literally adding the exponent into the floating point value exponent part.
* It is mostly precise, but it breaks on extremes.
*/

#pragma once

#include <cstdint>

// Two instruction exp2 for my use-case.
inline constexpr float exp2i(int32_t i) {
  union { float f = 1.f; int32_t i; } val;
  val.i += i << 23; // reinterpres as int, add i to exponent
  return val.f;
}
