#pragma once

#include "tree_volume.h"
#include "sampler_simd.h"
#include "blend_simd.h"

#include <ray/traversal_octree_simd.h>
#include <ray/intersection_simd.h>

#include <utils/utils.h>
#include <utils/simd.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename T, typename TransferFunctionType>
Vec4Vec render_tree(const TreeVolume<T> &volume, const RayVec &ray, float step, simd::float_m mask, const TransferFunctionType &transfer_function) {
  Vec4Vec dst = {0.f, 0.f, 0.f, 1.f};

  RayRangeVec range;

  intersect_aabb_ray(ray.origin, ray.direction_inverse, {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, range.min, range.max);

  mask = mask && range.min < range.max;

  if (mask.isNotEmpty()) {
    RayRangeVec slab_range { range.min, range.min };
    simd::float_v slab_start_value;

    ray_octree_traversal(ray, range, { }, 0, mask, [&](const RayRangeVec &range, const Vec3Vec &cell, uint32_t layer, simd::float_m &mask) {
      uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

      mask = mask && dst.a > 0.01f;

      if (mask.isEmpty()) {
        return;
      }

      // Next step does not intersect the block
      mask = mask && slab_range.max < range.max;

      if (mask.isEmpty()) {
        return;
      }

      Vec3Vec block = cell * simd::float_v(approx_exp2(layer));

      simd::float_m ray_outside = mask && ((block.x >= volume.info.layers[layer_index].width_in_blocks)
                                           || (block.y >= volume.info.layers[layer_index].height_in_blocks)
                                           || (block.z >= volume.info.layers[layer_index].depth_in_blocks));

      if (ray_outside.isNotEmpty()) {
        slab_range.min(ray_outside) = range.max;
        slab_range.max(ray_outside) = range.max;

        mask = mask && !ray_outside;

        if (mask.isEmpty()) {
          return;
        }
      }

      float stepsize = step * approx_exp2(layer_index);

      std::array<uint64_t, simd::len> node_handle;
      simd::float_v node_min;
      simd::float_v node_max;

      for (uint32_t k = 0; k < simd::len; k++) {
        if (mask[k]) {
          node_handle[k] = volume.info.node_handle(block[0][k], block[1][k], block[2][k], layer_index);
          node_min[k] = volume.nodes[node_handle[k]].min;
          node_max[k] = volume.nodes[node_handle[k]].max;
        }
      }

      Vec4Vec node_rgba = transfer_function(node_min, node_max, mask);

      simd::float_m block_empty = mask && (node_rgba.a == 0.f);

      // Empty space skipping
      if (block_empty.isNotEmpty()) {
        blend(transfer_function(slab_start_value, node_min, block_empty), dst, slab_range.max - slab_range.min, block_empty); // finish previous step with block value

        slab_range.min(block_empty) = range.max;
        slab_range.max(block_empty) = range.max;

        mask = mask && !block_empty;

        if (mask.isEmpty()) {
          return;
        }
      }

      simd::float_m node_uniform = mask && (node_min == node_max);

      // Fast integration of uniform space
      if (node_uniform.isNotEmpty()) {
        blend(transfer_function(slab_start_value, node_min, node_uniform), dst, slab_range.max - slab_range.min, node_uniform); // finish previous step with block value
        blend(node_rgba, dst, range.max - slab_range.max, node_uniform); // blend the rest of the block

        slab_start_value(node_uniform) = node_min;
        slab_range.min(node_uniform) = range.max;
        slab_range.max(node_uniform) = range.max + stepsize;

        mask = mask && !node_uniform;

        if (mask.isEmpty()) {
          return;
        }
      }

      // Recurse condition
      if (layer_index > 0) {
        return;
      }

      // Numeric integration
      for (mask = mask && (slab_range.max < range.max); mask.isNotEmpty(); mask = mask && (slab_range.max < range.max)) {
        Vec3Vec pos = ray.origin + ray.direction * slab_range.max;

        Vec3Vec in_block = (pos - cell) * simd::float_v(approx_exp2(layer)) * simd::float_v(TreeVolume<T>::SUBVOLUME_SIDE);

        simd::float_v slab_end_value;

        for (uint32_t k = 0; k < simd::len; k++) {
          if (mask[k]) {
            slab_end_value[k] = linterp(samplet(volume, volume.nodes[node_handle[k]].block_handle, in_block.x[k], in_block.y[k], in_block.z[k]));
          }
        }

        blend(transfer_function(slab_start_value, slab_end_value, mask), dst, slab_range.max - slab_range.min, mask);

        slab_start_value(mask) = slab_end_value;
        slab_range.min(mask) = slab_range.max;
        slab_range.max(mask) = slab_range.max + stepsize;
      }
    });
  }

  return dst;
}
