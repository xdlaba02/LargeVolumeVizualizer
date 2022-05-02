/**
* @file tree_slab.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Function that integrates a tree volume with a ray.
*/

#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler.h>

#include <ray/traversal_octree.h>
#include <ray/intersection.h>

#include <utils/fast_exp2.h>
#include <utils/blend.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename T, uint32_t N, typename F, typename P>
glm::vec4 integrate_tree_slab(const TreeVolume<T, N> &volume, const Ray &ray, float step, float terminate_thresh, const F &transfer_function, const P &integrate_predicate) {
  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  RayRange range = intersect_aabb_ray(ray, {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac});

  if (range.min < range.max) {

    // First slab in infinitely small because we want to initialize start value with first encountered value without producing output
    RayRange slab_range { range.min, range.min };
    float slab_start_value;

    ray_octree_traversal(ray, range, { 0.f, 0.f, 0.f }, 0, [&](const RayRange &range, const glm::vec3 &cell, uint32_t layer) {

      // Early ray termination
      if (dst.a < terminate_thresh) {
        return false;
      }

      // Next step does not intersect the block
      if (slab_range.max >= range.max) {
        return false;
      }

      uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

      glm::vec3 node_pos = cell * exp2i(layer);

      if (node_pos.x >= volume.info.layers[layer_index].width_in_nodes
      || (node_pos.y >= volume.info.layers[layer_index].height_in_nodes)
      || (node_pos.z >= volume.info.layers[layer_index].depth_in_nodes)) {
        slab_range.min = range.max;
        slab_range.max = range.max;
        return false;
      }

      const auto &node = volume.node(volume.info.node_handle(node_pos[0], node_pos[1], node_pos[2], layer_index));

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
        slab_range.max = range.max + step;

        return false;
      }

      if (integrate_predicate(dst.a, layer)) {

        float coef = exp2i(layer) * float(TreeVolume<T, N>::SUBVOLUME_SIDE);
        glm::vec3 shift = (ray.origin - cell) * coef;
        glm::vec3 mult = ray.direction * coef;

        // Numeric integration
        while (slab_range.max < range.max) {
          glm::vec3 in_block_pos = slab_range.max * mult + shift;

          float slab_end_value = sample(volume, node.block_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);

          blend(transfer_function(slab_start_value, slab_end_value), dst, slab_range.max - slab_range.min);

          slab_start_value = slab_end_value;
          slab_range.min = slab_range.max;
          slab_range.max = slab_range.max + step;

          // Early ray termination
          if (dst.a < terminate_thresh) {
            break;
          }
        }

        return false;
      }

      return true;
    });
  }

  return dst;
}
