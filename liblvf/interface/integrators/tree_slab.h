#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler.h>

#include <components/ray/traversal_octree.h>
#include <components/ray/intersection.h>

#include <components/fast_exp2.h>
#include <components/blend.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename T, typename TransferFunctionType, typename IntegratePredicate>
glm::vec4 integrate_tree_slab(const TreeVolume<T> &volume, const Ray &ray, float step, float terminate_thresh, const TransferFunctionType &transfer_function, const IntegratePredicate &integrate_predicate) {
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

      glm::vec3 block = cell * exp2i(layer);

      if (block.x >= volume.info.layers[layer_index].width_in_blocks
      || (block.y >= volume.info.layers[layer_index].height_in_blocks)
      || (block.z >= volume.info.layers[layer_index].depth_in_blocks)) {
        slab_range.min = range.max;
        slab_range.max = range.max;
        return false;
      }

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
        slab_range.max = range.max + step;

        return false;
      }

      if (integrate_predicate(cell, layer)) {
        // Numeric integration
        while (slab_range.max < range.max) {
          glm::vec3 pos = ray.origin + ray.direction * slab_range.max;

          glm::vec3 in_block = (pos - cell) * exp2i(layer) * float(TreeVolume<T>::SUBVOLUME_SIDE);

          float slab_end_value = sample(volume, node.block_handle, in_block.x, in_block.y, in_block.z);

          blend(transfer_function(slab_start_value, slab_end_value), dst, slab_range.max - slab_range.min);

          slab_start_value = slab_end_value;
          slab_range.min = slab_range.max;
          slab_range.max = slab_range.max + step;
        }

        return false;
      }

      return true;
    });
  }

  return dst;
}
