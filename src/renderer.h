
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
  Ray(const glm::vec3 &origin, const glm::vec3 &direction): origin(origin), direction(direction), direction_inverse(1.f / direction) {}

  const glm::vec3 origin;
  const glm::vec3 direction;
  const glm::vec3 direction_inverse;
};

// Interactive isosurface ray tracing of large octree volumes
// https://www.researchgate.net/publication/310054812_Interactive_isosurface_ray_tracing_of_large_octree_volumes
template <typename T, typename F>
void recursive_integrate(const Ray &ray, const RayRange &range, glm::vec<3, uint32_t> cell, uint32_t layer, F &callback) {
  if (callback(range, cell >> layer /* FIXME this might be off one bit error*/, layer)) {
    uint32_t child_bit = 1 << layer;

    glm::vec<3, uint32_t> center = cell | child_bit;

    glm::vec3 penter = ray.origin + ray.direction * range.min;

    glm::vec<3, uint32_t> child_cell = cell;

    child_cell.x |= penter.x > center.x ? child_bit : 0;
    child_cell.y |= penter.y > center.y ? child_bit : 0;
    child_cell.z |= penter.z > center.z ? child_bit : 0;

    glm::vec3 tcenter = (glm::vec3(center) - ray.origin) * ray.direction_inverse;

    // fast sort axis by tcenter
    std::array<uint8_t, 3> axis;

#if 1
    uint8_t order = ((tcenter[0] < tcenter[1]) << 2)
                  | ((tcenter[0] < tcenter[2]) << 1)
                  | ((tcenter[1] < tcenter[2]) << 0);

    axis[0] = std::array<uint8_t, 8>{ 2, 1, 0, 1, 2, 0, 0, 0 }[order];
    axis[1] = std::array<uint8_t, 8>{ 1, 2, 0, 0, 0, 1, 2, 1 }[order];
    axis[2] = std::array<uint8_t, 8>{ 0, 0, 0, 2, 1, 2, 1, 2 }[order];
#else
    bool gt01 = tcenter[0] > tcenter[1];
    bool gt02 = tcenter[0] > tcenter[2];
    bool gt12 = tcenter[1] > tcenter[2];

    axis[ gt01 +  gt02] = 0;
    axis[!gt01 +  gt12] = 1;
    axis[!gt02 + !gt12] = 2;
#endif

    float tmin = range.min;
    for (uint8_t i = 0; i < 3; i++) {
      float tmax = tcenter[axis[i]];

      if (tmin < tmax) {
        recursive_integrate(ray, { tmin, tmax }, child_cell, layer - 1, callback);
        tmin = tmax;

        child_cell[axis[i]] ^= child_bit;
      }
    }

    if (tmin < range.max) {
      recursive_integrate(ray, { tmin, range.max }, child_cell, layer - 1, callback);
    }
  }
}

template <typename T, typename F>
void render(const BlockedVolume<T> &volume, const glm::mat4 &mv, uint32_t width, uint32_t height, float yfov_degrees, const F &output) {

  glm::vec3 size_in_blocks { volume.info.layers[0].width_in_blocks,
                             volume.info.layers[0].height_in_blocks,
                             volume.info.layers[0].depth_in_blocks };

  // This transfroms ray from cannocical clip space to object space where volume is inside boundig box (0, 0, 0), (width_in_blocks, height_in_blocks, depth_in_blocks)
  glm::mat4 ray_transform = glm::scale(glm::mat4(1.f), size_in_blocks) * glm::translate(glm::mat4(1.f), glm::vec3(0.5f, 0.5f, 0.5f)) * glm::inverse(mv);

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

      float tmin {};
      float tmax {};

      intersect_aabb_ray(ray.origin, ray.direction_inverse, {}, size_in_blocks, tmin, tmax);

      glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

      if (tmin < tmax) {
        recursive_integrate(ray, { tmin, tmax }, { 0, 0, 0 }, std::size(volue.layers) - 1, /* TODO output */);
      }

      output(i, j, dst);
    }
  }
}

bool traverse(Ray ray, int depth, uint node_index, Vec3f cell, float tenter, float texit) {
  Vec3f center = Vec3f( cell | Vec3i(child_bit) );

  Vec3f tcenter = (center - ray.orig) / ray.dir;

  Vec3f penter = ray.orig + ray.dir * tenter;
  Vec3i child_cell = cell;
  Vec3i tc;

  tc.x = (penter.x >= center.x);
  tc.y = (penter.y >= center.y);
  tc.z = (penter.z >= center.z);

  int child = tc.x << 2 | tc.y << 1 | tc.z;

  child_cell.x |= tc.x ? child_bit : 0;
  child_cell.y |= tc.y ? child_bit : 0;
  child_cell.z |= tc.z ? child_bit : 0;

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
      child_cell[axis_isects[i]] &= ~child_bit;
    }
    else {
      child |= axisbit;
      child_cell[axis_isects[i]] |= child_bit;
    }
  }

  return false;
}
