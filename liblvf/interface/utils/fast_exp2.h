#pragma once

#include <cstdint>

// Two instruction exp2 for my use-case.
inline constexpr float exp2i(int32_t i) {
  union { float f = 1.f; int32_t i; } val;
  val.i += i << 23; // reinterpres as int, add i to exponent
  return val.f;
}
