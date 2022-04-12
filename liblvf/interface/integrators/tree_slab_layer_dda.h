#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler.h>

#include <ray/traversal_raster.h>
#include <ray/intersection.h>

#include <utils/fast_exp2.h>
#include <utils/blend.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, uint32_t N, typename F>
glm::vec4 integrate_tree_slab_layer_dda(const TreeVolume<T, N> &volume, const Ray &ray, uint8_t layer, float step, float terminate_thresh, const F &transfer_function) {
  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  RayRange range = intersect_aabb_ray(ray, {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac});

  uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

  if (range.min < range.max) {
    // First slab in infinitely small because we want to initialize start value with first encountered value without producing output
    RayRange slab_range { range.min, range.min };
    float slab_start_value {};

    Ray scaled_ray { ray.origin * exp2i(layer), ray.direction * exp2i(layer), 1.f / (ray.direction * exp2i(layer)) };

    ray_raster_traversal(scaled_ray, range, [&](const RayRange &range, const glm::vec<3, uint32_t> &node_pos) {

      // Early ray termination
      if (dst.a < terminate_thresh) {
        return false;
      }

      // Next step does not intersect the block
      if (slab_range.max >= range.max) {
        return true;
      }

      if (node_pos.x >= volume.info.layers[layer_index].width_in_nodes
      ||  node_pos.y >= volume.info.layers[layer_index].height_in_nodes
      ||  node_pos.z >= volume.info.layers[layer_index].depth_in_nodes) {
        slab_range.min = range.max;
        slab_range.max = range.max;
        return true;
      }

      const auto &node = volume.node(volume.info.node_handle(node_pos.x, node_pos.y, node_pos.z, layer_index));

      auto node_rgba = transfer_function(node.min, node.max);

      // Empty space skipping
      if (node_rgba.a == 0.f) {
        blend(transfer_function(slab_start_value, node.min), dst, slab_range.max - slab_range.min); // finish previous step with block value

        slab_range.min = range.max;
        slab_range.max = range.max;

        return true;
      }

      // Fast integration of uniform space
      if (node.min == node.max) {
        blend(transfer_function(slab_start_value, node.min), dst, slab_range.max - slab_range.min); // finish previous step with block value
        blend(node_rgba, dst, range.max - slab_range.max); // blend the rest of the block

        slab_start_value = node.min;
        slab_range.min = range.max;
        slab_range.max = range.max + step;

        return true;
      }

      // Numeric integration
      while (slab_range.max < range.max) {
        glm::vec3 pos = scaled_ray.origin + scaled_ray.direction * slab_range.max;

        glm::vec3 in_block_pos = (pos - glm::vec3(node_pos)) * float(TreeVolume<T, N>::SUBVOLUME_SIDE);

        float slab_end_value = sample(volume, node.block_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);

        glm::vec4 src = transfer_function(slab_start_value, slab_end_value);

        blend(src, dst, slab_range.max - slab_range.min);

        slab_start_value = slab_end_value;
        slab_range.min = slab_range.max;
        slab_range.max = slab_range.max + step;
      }

      return true;
    });
  }

  return dst;
}
