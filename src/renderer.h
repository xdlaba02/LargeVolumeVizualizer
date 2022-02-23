
#include "blocked_volume.h"
#include "intersection.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cstddef>

#include <numeric>

struct Ray {
  Ray(const glm::vec3 &origin, const glm::vec3 &direction)
      : origin(origin)
      , direction(direction)
      , direction_inverse(1.f / direction)
  {}

  glm::vec3 origin;
  glm::vec3 direction;
  glm::vec3 direction_inverse;
};

struct BoundingBox {
  glm::vec3 min;
  glm::vec3 max;
};

struct RayRange {
  float min;
  float max;
};

template <typename T, typename F>
void recursive_integrate(const BlockedVolume<T> &volume, const Ray &ray, const RayRange &range, const BoundingBox &bb, uint32_t layer, F &callback) {
  if (layer == std::size(volume.layers) - 1) {

    float stepsize = m_stepsize * (std::size(volume.layers) - layer);

    for (float t = range.min; t < range.max; t += stepsize) {
      glm::vec3 sample = ray.origin + ray.direction * tmin;

      callback(volume.sample_volume(sample.x, sample.y, sample.z, layer));
    }
  }
  else {
    glm::vec3 avg = (bb.min + bb.max) / 2.f;

    BoundingBox bbs[8];
    RayRange ranges[8];

    std::array<const glm::vec3 *, 3> points {&bb.min, &avg, &bb.max};

    // TODO inline and optimize
    for (uint8_t z: {0, 1}) {
      for (uint8_t y: {0, 1}) {
        for (uint8_t x: {0, 1}) {
          uint8_t i = z << 2 | y << 2 | x;

          bbs[i].min = {points[x + 0]->x, points[y + 0]->y, points[z + 0]->z};
          bbs[i].max = {points[x + 1]->x, points[y + 1]->y, points[z + 1]->z};

          intersect_aabb_ray(ray.origin, ray.direction_inverse, bbs[i].min, bbs[i].max, ranges[i].min, ranges[i].max);
        }
      }
    }

    uint8_t octants[8];
    std::iota(std::begin(octants), std::end(octants), 0);

    // Sort first fout octants from nearest to furthest - the ray cannot collide with more than 4 octants
    std::partial_sort(std::begin(octants), std::begin(octants) + 4, std::end(octants), [&](uint8_t l, uint8_t r) { return ranges[l].min < ranges[r].max; });

    for (uint8_t i: {0, 1, 2, 3}) {
      if (ranges[octants[i]].min < tmax) {
        recursive_integrate(volume, ray, ranges[octants[i]], bbs[octants[i]], layer + 1, dst);
      }
    }
  }
}

template <typename T, typename F>
void render(const BlockedVolume<T> &volume, const glm::mat4 &mv, uint32_t width, uint32_t height, float yfov_degrees, const F &output) {

  // This transfroms ray from cannocical clip space to normalized object space where volume is inside boundig box (0, 0, 0), (1, 1, 1)
  glm::mat4 ray_transform = glm::translate(glm::mat4(1.f), glm::vec3(0.5f, 0.5f, 0.5f)) * glm::inverse(mv);

  glm::vec3 origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

  float yfov = glm::radians(yfov_degrees);
  float yfov_coef = std::tan(yfov / 2.f);
  float xfov_coef = yfov * width / height;

  float width_coef_avg  = xfov_coef / width;
  float width_coef  = 2.f * width_coef_avg;
  float width_shift = width_coef_avg - xfov_coef;

  float height_coef_avg = yfov_coef / height;
  float height_coef = 2.f * height_coef_avg;
  float height_shift = height_coef_avg - yfov_coef;

  for (uint32_t j = 0; j < height; j++) {
    float y = j * height_coef + height_shift;

    for (uint32_t i = 0; i < width; i++) {
      float x = i * width_coef + width_shift;

      Ray ray(origin, ray_transform * glm::normalize(glm::vec4(x, y, -1.f, 0.f)));

      float tmin, tmax;

      intersect_aabb_ray(ray.origin, ray.direction_inverse, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, tmin, tmax);

      glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

      if (tmin < tmax) {
        recursive_integrate(volume, ray, tmin, tmax, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, 0, dst);
      }

      output(i, j, dst);
    }
  }
}
