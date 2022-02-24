
#include "blocked_volume.h"
#include "intersection.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cstddef>

#include <numeric>

struct BoundingBox {
  glm::vec3 min;
  glm::vec3 max;
};

struct RayRange {
  float min = std::numeric_limits<float>::infinity();
  float max = -std::numeric_limits<float>::infinity();
};

struct Ray {
  Ray(const glm::vec3 &origin, const glm::vec3 &direction)
      : origin(origin)
      , direction(direction)
      , direction_inverse(1.f / direction)
  {
    std::array<float, 8> tmins {};

    static constinit std::array<glm::vec3, 3> points { glm::vec3{ 0.0f, 0.0f, 0.0f },
                                                       glm::vec3{ 0.5f, 0.5f, 0.5f },
                                                       glm::vec3{ 1.0f, 1.0f, 1.0f } };

    for (uint8_t z: {0, 1}) {
      for (uint8_t y: {0, 1}) {
        for (uint8_t x: {0, 1}) {
          uint8_t i = z << 2 | y << 1 | x;

          glm::vec3 min = { points[x + 0].x, points[y + 0].y, points[z + 0].z };
          glm::vec3 max = { points[x + 1].x, points[y + 1].y, points[z + 1].z };

          float dummy_max;
          intersect_aabb_ray(origin, direction_inverse, min, max, tmins[i], dummy_max);
        }
      }
    }

    std::iota(std::begin(octant_order), std::end(octant_order), 0);
    std::sort(std::begin(octant_order), std::end(octant_order), [&](uint8_t l, uint8_t r) { return tmins[l] < tmins[r]; });
  }

  glm::vec3 origin;
  glm::vec3 direction;
  glm::vec3 direction_inverse;
  std::array<uint8_t, 8> octant_order;
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

    BoundingBox bbs[2][2][2];
    RayRange ranges[0][1][0][2][2];

    bbs[0][0][0].min = {bb.min.x, bb.min.y, bb.min.z};
    bbs[0][0][0].max = {avg.x, avg.y, avg.z};

    bbs[0][0][1].min = {avg.x, bb.min.y, bb.min.z};
    bbs[0][0][1].max = {bb.max.x, avg.y, avg.z};

    bbs[0][1][0].min = {bb.min.x, avg.y, bb.min.z};
    bbs[0][1][0].max = {avg.x, bb.max.y, avg.z};

    bbs[0][1][1].min = {avg.x, avg.y, bb.min.z};
    bbs[0][1][1].max = {bb.max.x, bb.max.y, avg.z};

    bbs[1][0][0].min = {bb.min.x, bb.min.y, avg.z};
    bbs[1][0][0].max = {avg.x, avg.y, bb.max.z};

    bbs[1][0][1].min = {avg.x, bb.min.y, avg.z};
    bbs[1][0][1].max = {bb.max.x, avg.y, bb.max.z};

    bbs[1][1][0].min = {bb.min.x, avg.y, avg.z};
    bbs[1][1][0].max = {avg.x, bb.max.y, bb.max.z};

    bbs[1][1][1].min = {avg.x, avg.y, avg.z};
    bbs[1][1][1].max = {bb.max.x, bb.max.y, bb.max.z};

    glm::vec3 tavg = (avg - ray.origin) * ray.direction_inverse;

    if (ray.direction_inverse.x < 0.f) {
      for (uint8_t z: {0, 1}) {
        for (uint8_t y: {0, 1}) {
          ranges[z][y][0].min = std::max(ranges[z][y][0].min, tavg.x);
          ranges[z][y][1].max = std::min(ranges[z][y][1].max, tavg.x);
        }
      }
    }
    else {
      for (uint8_t z: {0, 1}) {
        for (uint8_t y: {0, 1}) {
          ranges[z][y][1].min = std::max(ranges[z][y][1].min, tavg.x);
          ranges[z][y][0].max = std::min(ranges[z][y][0].max, tavg.x);
        }
      }
    }

    if (ray.direction_inverse.y < 0.f) {
      for (uint8_t z: {0, 1}) {
        for (uint8_t x: {0, 1}) {
          ranges[z][0][x].min = std::max(ranges[z][0][x].min, tavg.y);
          ranges[z][1][x].max = std::min(ranges[z][1][x].max, tavg.y);
        }
      }
    }
    else {
      for (uint8_t z: {0, 1}) {
        for (uint8_t x: {0, 1}) {
          ranges[z][1][x].min = std::max(ranges[z][1][x].min, tavg.y);
          ranges[z][0][x].max = std::min(ranges[z][0][x].max, tavg.y);
        }
      }
    }

    if (ray.direction_inverse.z < 0.f) {
      for (uint8_t y: {0, 1}) {
        for (uint8_t x: {0, 1}) {
          ranges[0][y][x].min = std::max(ranges[0][y][x].min, tavg.z);
          ranges[1][y][x].max = std::min(ranges[1][y][x].max, tavg.z);
        }
      }
    }
    else {
      for (uint8_t y: {0, 1}) {
        for (uint8_t x: {0, 1}) {
          ranges[1][y][x].min = std::max(ranges[1][y][x].min, tavg.z);
          ranges[0][y][x].max = std::min(ranges[0][y][x].max, tavg.z);
        }
      }
    }

    uint8_t octants[8];
    std::iota(std::begin(octants), std::end(octants), 0);

    // Sort first fout octants from nearest to furthest - the ray cannot collide with more than 4 octants
    std::partial_sort(std::begin(octants), std::begin(octants) + 4, std::end(octants), [&](uint8_t l, uint8_t r) { return ranges[l].min < ranges[r].min; });

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
