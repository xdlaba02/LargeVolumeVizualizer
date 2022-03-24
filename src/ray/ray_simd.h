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
