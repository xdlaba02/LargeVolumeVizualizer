#include "../external/glm/glm/glm.hpp"

#include <array>
#include <iostream>

inline void intersect_aabb_ray(const glm::vec3& origin, glm::vec3 ray_direction_inverse, const glm::vec3 &min, const glm::vec3 &max, float& tmin, float& tmax) {
  tmin = -std::numeric_limits<float>::infinity();
  tmax = std::numeric_limits<float>::infinity();

  for (uint32_t i = 0; i < 3; ++i) {
    float t0 = (min[i] - origin[i]) * ray_direction_inverse[i];
    float t1 = (max[i] - origin[i]) * ray_direction_inverse[i];

    if (ray_direction_inverse[i] < 0.f) {
      tmin = std::max(tmin, t1);
      tmax = std::min(tmax, t0);
    }
    else {
      tmin = std::max(tmin, t0);
      tmax = std::min(tmax, t1);
    }
  }

  tmin = std::max(tmin, 0.f); // solves when ray hits box from behind
}

struct Ray {
  Ray(const glm::vec3 &origin, const glm::vec3 &direction)
      : origin(origin)
      , direction(direction)
      , direction_inverse(1.f / direction) {}

  const glm::vec3 origin;
  const glm::vec3 direction;
  const glm::vec3 direction_inverse;
};


struct RayRange {
  float min;
  float max;
};

// Two instruction exp2 for my use-case.
float approx_exp2(int32_t i) {
  union { float f = 1.f; int32_t i; } val;
  val.i += i << 23; // reinterpres as int, add i to exponent
  return val.f;
}

template <typename F>
void ray_octree_traversal(const Ray &ray, const RayRange &range, const glm::vec3 &cell, uint32_t layer, const F &callback) {
  if (callback(range, cell, layer)) {
    // TODO precompute?
    float child_size = approx_exp2(-layer - 1);

    glm::vec3 center = cell + child_size;

    glm::vec3 tcenter = (center - ray.origin) * ray.direction_inverse;

    // fast sort axis by tcenter
    std::array<uint8_t, 3> axis;

    bool gt01 = tcenter[0] > tcenter[1];
    bool gt02 = tcenter[0] > tcenter[2];
    bool gt12 = tcenter[1] > tcenter[2];

    axis[ gt01 +  gt02] = 0;
    axis[!gt01 +  gt12] = 1;
    axis[!gt02 + !gt12] = 2;

    glm::vec3 child_cell = cell;
    glm::vec3 opposite_cell = center;

    // TODO precompute?
    // FIXME is this right?
    for (uint8_t i = 0; i < 3; i++) {
      if (ray.direction[i] < 0.f) {
        std::swap(child_cell[i], opposite_cell[i]);
      }
    }

    float tmin = range.min;

    for (uint8_t i = 0; i < 3; i++) {
      float tmax = std::min(tcenter[axis[i]], range.max);

      if (tmin < tmax) {
        ray_octree_traversal(ray, { tmin, tmax }, child_cell, layer + 1, callback);
        tmin = tmax;
      }

      // outside the condition because the way of swapping child cells by direction
      // FIXME is this right?
      child_cell[axis[i]] = opposite_cell[axis[i]];
    }

    if (tmin < range.max) {
      ray_octree_traversal(ray, { tmin, range.max }, child_cell, layer + 1, callback);
    }
  }
}

int main(void) {
  glm::vec3 origin { 0.25f, 0.f, 0.f };
  glm::vec3 direction { .5f, 1.f, 0.f };

  Ray ray(origin, direction);

  float tmin {};
  float tmax {};

  intersect_aabb_ray(ray.origin, ray.direction_inverse, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, tmin, tmax);

  ray_octree_traversal(ray, { tmin, tmax }, {0, 0, 0}, 0, [&](const RayRange &range, const glm::vec3 &cell, uint32_t layer) {
    auto block = cell * approx_exp2(layer);

    auto min = ray.origin + ray.direction * range.min;

    auto start = (min * approx_exp2(layer) - block) * 15.f - 0.5f;

    std::cerr << std::string(layer * 2, ' ') << "layer " << layer << ": range " << range.min << " " << range.max << ", block " << block.x << " " << block.y << ", start " << start.x << " " << start.y << "\n";

    return layer < 4;
  });
}
