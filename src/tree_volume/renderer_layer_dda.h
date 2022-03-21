#pragma once

#include "tree_volume.h"
#include "sampler.h"
#include "blend.h"

#include <ray/traversal_raster.h>
#include <ray/intersection.h>

#include <utils/utils.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
glm::vec4 render_layer_dda(const TreeVolume<T> &volume, const Ray &ray, uint8_t layer, float step, const TransferFunctionType &transfer_function) {
  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  RayRange range {};

  intersect_aabb_ray(ray.origin, ray.direction_inverse, {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, range.min, range.max);

  if (range.min < range.max) {
    float stepsize = step * approx_exp2(layer);

    // First slab in infinitely small because we want to initialize start value with first encountered value without producing output
    RayRange slab_range { range.min, range.min };
    float slab_start_value {};

    uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

    Ray scaled_ray { ray.origin * approx_exp2(layer_index), ray.direction * approx_exp2(layer_index), 1.f / (ray.direction * approx_exp2(layer_index)) };

    ray_raster_traversal(scaled_ray, range, [&](const RayRange &range, const glm::vec<3, uint32_t> &block) {

      // Early ray termination
      if (dst.a < 0.01f) {
        return false;
      }

      // Next step does not intersect the block
      if (slab_range.max >= range.max) {
        return true;
      }

      if (block.x >= volume.info.layers[layer].width_in_blocks
      ||  block.y >= volume.info.layers[layer].height_in_blocks
      ||  block.z >= volume.info.layers[layer].depth_in_blocks) {
        slab_range.min = range.max;
        slab_range.max = range.max;
        return true;
      }

      const auto &node = volume.nodes[volume.info.node_handle(block.x, block.y, block.z, layer)];

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
        slab_range.max = range.max + stepsize;

        return true;
      }

      // Numeric integration
      while (slab_range.max < range.max) {
        glm::vec3 pos = scaled_ray.origin + scaled_ray.direction * slab_range.max;

        glm::vec3 in_block = (pos - glm::vec3(block)) * float(TreeVolume<T>::SUBVOLUME_SIDE);

        float slab_end_value = linterp(samplet(volume, node.block_handle, in_block.x, in_block.y, in_block.z));

        glm::vec4 src = transfer_function(slab_start_value, slab_end_value);

        blend(src, dst, slab_range.max - slab_range.min);

        slab_start_value = slab_end_value;
        slab_range.min = slab_range.max;
        slab_range.max = slab_range.max + stepsize;
      }

      return true;
    });
  }

  return dst;
}
