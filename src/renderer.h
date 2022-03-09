
#include "blocked_volume.h"
#include "intersection.h"
#include "ray.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cstddef>
#include <cstdint>

#include <numeric>
#include <array>

struct RayRange {
  float min;
  float max;
};

// Two instruction exp2 for my use-case.
constexpr float approx_exp2(int32_t i) {
  union { float f = 1.f; int32_t i; } val;
  val.i += i << 23; // reinterpres as int, add i to exponent
  return val.f;
}



// TODO intersection with filled part only
// TODO Phong shading
// TODO Determinace uzlu
// TODO Ray packlet + simd
// TODO Rozhrani
// TODO Precomputes?

// Interactive isosurface ray tracing of large octree volumes
// https://www.researchgate.net/publication/310054812_Interactive_isosurface_ray_tracing_of_large_octree_volumes
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
      child_cell[axis[i]] = opposite_cell[axis[i]];
    }

    if (tmin < range.max) {
      ray_octree_traversal(ray, { tmin, range.max }, child_cell, layer + 1, callback);
    }
  }
}

void integrate(const glm::vec4 &src, glm::vec4 &dst, float stepsize) {
  float alpha = 1.f - std::exp(-src.a * stepsize);

  float coef = alpha * dst.a;

  dst.r += src.r * coef;
  dst.g += src.g * coef;
  dst.b += src.b * coef;
  dst.a *= 1.f - alpha;
};

template <typename T, typename TransferFunctionType>
glm::vec4 render(const BlockedVolume<T> &volume, const Ray &ray, float step, const TransferFunctionType &transfer_function) {
  float tmin {};
  float tmax {};

  // TODO intersection with filled part only
  intersect_aabb_ray(ray.origin, ray.direction_inverse, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, tmin, tmax);

  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  if (tmin < tmax) {

    // First slab in infinitely small because we want to initialize start value with first encountered value without producing output
    float slab_start_t = tmin;
    float slab_end_t = tmin;
    float slab_start_value = 0.f;

    ray_octree_traversal(ray, { tmin, tmax }, { 0.f, 0.f, 0.f }, 0, [&](const RayRange &range, const glm::vec3 &cell, uint32_t layer) {

      // Early ray termination
      if (dst.a < 0.01f) {
        return false;
      }

      // Next step does not intersect the block
      if (slab_end_t >= range.max) {
        return false;
      }

      uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

      glm::vec3 layer_size {
        volume.info.layers[layer_index].width_in_blocks,
        volume.info.layers[layer_index].height_in_blocks,
        volume.info.layers[layer_index].depth_in_blocks
      };

      glm::vec3 block = cell * approx_exp2(layer);

      for (uint8_t i = 0; i < 3; i++) {
        // Octree block is outside the real volume
        if (block[i] >= layer_size[i]) {
          slab_start_t = range.max;
          slab_end_t = range.max;
          return false;
        }
      }

      const auto &node = volume.nodes[volume.info.node_handle(block[0], block[1], block[2], layer_index)];

      auto rgba = transfer_function(node.min, node.max);

      // Empty space skipping
      if (rgba.a == 0.f) {
        integrate(transfer_function(slab_start_value, node.min), dst, slab_end_t - slab_start_t); // finish previous step with block value

        slab_start_t = range.max;
        slab_end_t = range.max;

        return false;
      }

      // Fast integration of uniform space
      if (node.min == node.max) {
        integrate(transfer_function(slab_start_value, node.min), dst, slab_end_t - slab_start_t); // finish previous step with block value
        integrate(rgba, dst, range.max - slab_end_t); // integrate the rest of the block

        slab_start_value = node.min;
        slab_start_t = range.max;
        slab_end_t = range.max + step;

        return false;
      }

      // Recurse condition
      if (layer_index > 0) {
        return true;
      }

      // Laplacian integration
      while (slab_end_t < range.max) {
        glm::vec3 pos = ray.origin + ray.direction * slab_end_t;

        glm::vec3 in_block = (pos - cell) * approx_exp2(layer) * float(BlockedVolume<T>::SUBVOLUME_SIDE);

        float slab_end_value = volume.sample_block(node.block_handle, in_block.x, in_block.y, in_block.z);

        integrate(transfer_function(slab_start_value, slab_end_value), dst, slab_end_t - slab_start_t);

        slab_start_value = slab_end_value;
        slab_start_t = slab_end_t;
        slab_end_t = slab_end_t + step;
      }

      return false;
    });
  }

  return dst;
}
