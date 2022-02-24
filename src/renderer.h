
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

          tmins[i] = -std::numeric_limits<float>::infinity();

          for (uint32_t i = 0; i < 3; ++i) {
            float t0 = (min[i] - origin[i]) * direction_inverse[i];
            float t1 = (max[i] - origin[i]) * direction_inverse[i];

            if (ray_direction_inverse[i] < 0.f) {
              tmin = std::max(tmin, t1);
              tmax = std::min(tmax, t0);
            }
            else {
              tmin = std::max(tmin, t0);
              tmax = std::min(tmax, t1);
            }
          }
        }
      }
    }

    std::iota(std::begin(octant_order), std::end(octant_order), 0);
    std::sort(std::begin(octant_order), std::end(octant_order), [&](uint8_t l, uint8_t r) { return tmins[l] < tmins[r]; });
  }

  RayRange reduceRange(uint8_t octant, const RayRange &range, const glm::vec3 &tcenter) const {
    Range octant_range = range;

    for (uint8_t d: {0, 1, 2}) {
      usint8_t pos = octant_order[octant] & (1 << d);
      uint8_t near = direction[d] < 0;

      if (pos == near) {
        octant_range.max = std::min(octant_range.max, tcenter[d]);
      }
      else {
        octant_range.min = std::max(octant_range.min, tcenter[d]);
      }
    }

    return octant_range;
  }

  glm::vec3 origin;
  glm::vec3 direction;
  glm::vec3 direction_inverse;

private:
  std::array<uint8_t, 8> octant_order;
};

template <typename T, typename F>
void recursive_integrate(const BlockedVolume<T> &volume, const Ray &ray, const RayRange &range, const glm::vec3 &center, uint32_t layer, F &callback) {
  if (layer == std::size(volume.layers) - 1) {

    float stepsize = m_stepsize * (std::size(volume.layers) - layer);

    for (float t = range.min; t < range.max; t += stepsize) {
      glm::vec3 sample = ray.origin + ray.direction * tmin;

      callback(volume.sample_volume(sample.x, sample.y, sample.z, layer));
    }
  }
  else {
    // find collisions of the ray with three center planes
    glm::vec3 tcenter = (center - ray.origin) * ray.direction_inverse;

    // Find ranges of t for suboctants, recurse if intersecting.
    for (uint8_t i = 0; i < 8; i++) {
      // This is faster than ray-bb intersection. It reduces existing range.
      Range octant_range = ray.reduceRange(i, range, tcenter);

      if (octant_range.min < octant_range.max) {
        recursive_integrate(volume, ray, octant_range, /*TODO*/, layer + 1, dst);
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

      RayRange range {};

      intersect_aabb_ray(ray.origin, ray.direction_inverse, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, range.min, range.max);

      glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

      if (tmin < tmax) {
        recursive_integrate(volume, ray, range, {0.5f, 0.5f, 0.5f}, 0, dst);
      }

      output(i, j, dst);
    }
  }
}

bool traverse(Ray ray, int depth, uint node_index, Vec3f cell, float tenter, float texit) {
  Vec3f center = Vec3f( cell | Vec3i((1 << (max_depth - depth - 1))) );

  Vec3f tcenter = (center - ray.orig) / ray.dir;

  Vec3f penter = ray.orig + ray.dir * tenter;
  Vec3i child_cell = cell;
  Vec3i tc;

  tc.x = (penter.x >= center.x);
  tc.y = (penter.y >= center.y);
  tc.z = (penter.z >= center.z);

  int child = tc.x << 2 | tc.y << 1 | tc.z;

  child_cell.x |= tc.x ? (1 << (max_depth - depth - 1)) : 0;
  child_cell.y |= tc.y ? (1 << (max_depth - depth - 1)) : 0;
  child_cell.z |= tc.z ? (1 << (max_depth - depth - 1)) : 0;

  Vec3i axis_isects;

  {perform 3-way minimum of tcenter such that axis_isects
  contains the sorted intersection with the X,Y,Z
  octant mid-planes}

  const int axis_table[] = {4, 2, 1};
  float child_tenter = tenter;
  float child_texit;

  for({all valid axis_isects[i] while tcenter < texit} ; i++) {
    child_texit = min(tcenter[axis_isects[i]], texit);

    OctNode &node = nodes[depth][node_index];
    if (isovalue >= node.child_mins[child] || isovalue <= node.child_maxs[child]) {
      //traverse scalar leaf, cap or node
      if (node.child_offset == -1) {
        if (traverse_scalar_leaf(...)) {
          return true;
        }
      }
      else if (depth == max_depth - 2) {
        if (traverse_cap(...)) {
          return true;
        }
      }
      else {
        if (traverse(ray,depth + 1, child_cell, child_tenter, child_texit)) {
          return true;
        }
      }
    }

    if (child_texit == texit) {
      return false;
    }

    child_tenter = child_texit;
    axisbit = axis_table[axis_isects[i]];

    if (child & axisbit) {
      child &= ~axisbit;
      child_cell[axis_isects[i]] &= ~(1 << (max_depth - depth - 1));
    }
    else {
      child |= axisbit;
      child_cell[axis_isects[i]] |= (1 << (max_depth - depth - 1));
    }
  }

  return false;
}
