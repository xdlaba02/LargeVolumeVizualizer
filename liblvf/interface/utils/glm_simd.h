/**
* @file glm_simd.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Vectorized glm types that are used trough the library.
*/

#pragma once

#include "simd.h"

#include <glm/glm.hpp>

namespace simd {
  using vec3 = glm::vec<3, float_v>;
  using vec4 = glm::vec<4, float_v>;
}
