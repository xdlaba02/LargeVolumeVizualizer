#pragma once

#include <glm/glm.hpp>
#include <utils/simd.h>

using Vec3Vec = glm::vec<3, simd::float_v>;
using Vec4Vec = glm::vec<4, simd::float_v>;

struct RayVec {
  Vec3Vec origin;
  Vec3Vec direction;
  Vec3Vec direction_inverse;
};

struct RayRangeVec {
  simd::float_v min;
  simd::float_v max;
};
