/**
* @file ray_simd.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Vector data structures used for ray operations.
*/

#pragma once

#include <utils/glm_simd.h>

namespace simd {
  struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 direction_inverse;
  };

  struct RayRange {
    float_v min;
    float_v max;
  };

}
