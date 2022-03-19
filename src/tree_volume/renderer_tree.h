#pragma once

#include "tree_volume.h"
#include "sampler.h"
#include "blend.h"

#include <ray/traversal_octree.h>
#include <ray/intersection.h>

#include <utils/utils.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename T, typename TransferFunctionType>
glm::vec4 render_tree(const TreeVolume<T> &volume, const Ray &ray, float step, const TransferFunctionType &transfer_function) {
  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  RayRange range {};

  intersect_aabb_ray(ray.origin, ray.direction_inverse, {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, range.min, range.max);


  if (range.min < range.max) {

    // First slab in infinitely small because we want to initialize start value with first encountered value without producing output
    RayRange slab_range { range.min, range.min };
    float slab_start_value;

    ray_octree_traversal(ray, range, { 0.f, 0.f, 0.f }, 0, [&](const RayRange &range, const glm::vec3 &cell, uint32_t layer) {

      // Early ray termination
      if (dst.a < 0.01f) {
        return false;
      }

      // Next step does not intersect the block
      if (slab_range.max >= range.max) {
        return false;
      }

      uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

      glm::vec3 block = cell * approx_exp2(layer);

      if (block.x >= volume.info.layers[layer_index].width_in_blocks
      || (block.y >= volume.info.layers[layer_index].height_in_blocks)
      || (block.z >= volume.info.layers[layer_index].depth_in_blocks)) {
        slab_range.min = range.max;
        slab_range.max = range.max;
        return false;
      }

      // TODO precompute
      float stepsize = step * approx_exp2(layer_index);

      const auto &node = volume.nodes[volume.info.node_handle(block[0], block[1], block[2], layer_index)];

      auto node_rgba = transfer_function(node.min, node.max);

      // Empty space skipping
      if (node_rgba.a == 0.f) {
        blend(transfer_function(slab_start_value, node.min), dst, slab_range.max - slab_range.min); // finish previous step with block value

        slab_range.min = range.max;
        slab_range.max = range.max;

        return false;
      }

      // Fast integration of uniform space
      if (node.min == node.max) {
        blend(transfer_function(slab_start_value, node.min), dst, slab_range.max - slab_range.min); // finish previous step with block value
        blend(node_rgba, dst, range.max - slab_range.max); // blend the rest of the block

        slab_start_value = node.min;
        slab_range.min = range.max;
        slab_range.max = range.max + stepsize;

        return false;
      }

      // Recurse condition
      if (layer_index > 0) {
        return true;
      }

      // Numeric integration
      while (slab_range.max < range.max) {
        glm::vec3 pos = ray.origin + ray.direction * slab_range.max;

        glm::vec3 in_block = (pos - cell) * approx_exp2(layer) * float(TreeVolume<T>::SUBVOLUME_SIDE);

        Samplet sampl = samplet(volume, node.block_handle, in_block.x, in_block.y, in_block.z);

        float slab_end_value = linterp(sampl);

        glm::vec4 src = transfer_function(slab_start_value, slab_end_value);

#if 0
        std::array<float, 3> grad = gradient(sampl);
        if (grad[0] || grad[1] || grad[2]) {
          glm::vec3 gradient_normal = glm::normalize(glm::vec3{grad[0], grad[1], grad[2]});

          glm::vec3 light_direction = glm::normalize(glm::vec3{5, 5, 5} - pos);
          glm::vec3 halfway_vector  = glm::normalize(ray.direction + light_direction);

          float dot_diff = glm::max(0.f, glm::dot(gradient_normal, light_direction));

          float dot_spec = glm::max(0.f, glm::dot(halfway_vector, gradient_normal));

          static constinit float ka = 0.8f;
          static constinit float kd = 1.0f;
          static constinit float ks = 0.0f;
          static constinit float shininess = 1000.0f;

          float diffuse = kd * dot_diff;
          float specular = ks * std::pow(dot_spec, shininess);

          src = glm::vec4(glm::clamp(glm::vec3(src) * (ka + diffuse) + specular, 0.f, 1.f), src.a);
        }

#endif

        blend(src, dst, slab_range.max - slab_range.min);

        slab_start_value = slab_end_value;
        slab_range.min = slab_range.max;
        slab_range.max = slab_range.max + stepsize;
      }

      return false;
    });
  }

  return dst;
}
