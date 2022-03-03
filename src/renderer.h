
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

template <typename T, typename TransferFunctionType>
glm::vec4 render(const BlockedVolume<T> &volume, const glm::vec3 &ray_origin, const glm::vec3 &ray_direction, float step, const TransferFunctionType &transfer_function) {
  Ray ray(ray_origin + 0.5f, ray_direction);

  float tmin {};
  float tmax {};

  intersect_aabb_ray(ray.origin, ray.direction_inverse, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, tmin, tmax);

  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  auto integrate = [&](const glm::vec4 &rgba, float stepsize) {
    float alpha = 1.f - std::exp(-rgba.a * stepsize);

    float coef = alpha * dst.a;

    dst.r += rgba.r * coef;
    dst.g += rgba.g * coef;
    dst.b += rgba.b * coef;
    dst.a *= 1.f - alpha;
  };

  if (tmin < tmax) {
    float t = tmin;
    float next_t = tmin;
    float value = 0.f;

    ray_octree_traversal(ray, { tmin, tmax }, { 0.f, 0.f, 0.f }, 0, [&](const RayRange &range, const glm::vec3 &cell, uint32_t layer) {
      if (next_t >= range.max) {
        // The intersection is so small it can be skipped
        //return false;
      }

      glm::vec3 block = cell * approx_exp2(layer);

      uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

      glm::vec3 layer_size {
        volume.info.layers[layer_index].width_in_blocks,
        volume.info.layers[layer_index].height_in_blocks,
        volume.info.layers[layer_index].depth_in_blocks
      };

      for (uint8_t i = 0; i < 3; i++) {
        if (block[i] >= layer_size[i]) {
          // The block is outside the real volume
          return false;
        }
      }

      const auto &node = volume.nodes[volume.info.node_handle(block[0], block[1], block[2], layer_index)];

      auto node_rgba = transfer_function(node.min, node.max);

      if (node_rgba.a == 0.f) {
        // node is empty with current transfer function, skip
        return false;
      }

      if (node.min == node.max) {
        // fast integration
        integrate(transfer_function(value, node.min), next_t - t); // finish previous step with block value
        integrate(node_rgba, range.max - next_t); // integrate the rest of the block

        value = node.min;
        t = range.max;
        next_t = t + step;
      }
      else {
        // integration by sampling

        if (layer_index) {
          // dumb condition
          // TODO do something elaborate here
          return true;
        }

        while (next_t < range.max) {
          // FIXME something weird is happening here
          // 0..1 is in octree space. Convert to volume space (which is subset)

          glm::vec3 pos = ray.origin + ray.direction * next_t;

          //glm::vec3 in_block_denorm = (pos - cell) * layer_size * float(BlockedVolume<T>::SUBVOLUME_SIDE) - 0.5f;
          float next_value = volume.sample_volume(pos.x, pos.y, pos.z, layer_index);

          integrate(transfer_function(value, next_value), next_t - t);

          value = next_value;
          t = next_t;
          next_t = t + step;
        }
      }

      return false;
    });
  }

  return dst;
}
