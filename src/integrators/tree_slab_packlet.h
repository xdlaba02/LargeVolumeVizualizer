#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler_simd.h>

#include <ray/traversal_octree_packlet.h>
#include <ray/intersection.h>

#include <utils/utils.h>
#include <utils/simd.h>
#include <utils/blend_simd.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename T, typename TransferFunctionType, typename IntegratePredicate>
Vec4Packlet integrate_tree_slab_packlet(const TreeVolume<T> &volume, const RayPacklet &ray, float step, float terminate_thresh, MaskPacklet mask, const TransferFunctionType &transfer_function, const IntegratePredicate &integrate_predicate) {
  Vec4Packlet dst;

  RayRangePacklet range;
  RayRangePacklet slab_range;
  FloatPacklet slab_start_value;

  bool packlet_not_empty = false;
  for (uint32_t j = 0; j < simd::len; j++) {

    if (mask[j].isNotEmpty()) {
      range[j] = intersect_aabb_ray(ray[j], {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac});

      mask[j] = mask[j] && range[j].min < range[j].max;
      dst[j] = {0.f, 0.f, 0.f, 1.f};

      if (mask[j].isNotEmpty()) {
        packlet_not_empty = true;
        slab_range[j] = { range[j].min, range[j].min };
      }
    }
  }

  if (packlet_not_empty) {
    ray_octree_traversal(ray, range, { }, 0, mask, [&](const RayRangePacklet &range, const Vec3Packlet &cell, uint8_t layer, MaskPacklet &mask) {
      uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

      Vec3Packlet block;
      float stepsize = step / exp2i(layer);

      for (uint32_t j = 0; j < simd::len; j++) {
        if (mask[j].isEmpty()) {
          continue;
        }

        mask[j] = mask[j] && dst[j].a > terminate_thresh;

        if (mask[j].isEmpty()) {
          continue;
        }

        // Next step does not intersect the block
        mask[j] = mask[j] && slab_range[j].max < range[j].max;

        if (mask[j].isEmpty()) {
          continue;
        }

        block[j] = cell[j] * simd::float_v(exp2i(layer));

        simd::float_m ray_outside = mask[j] && ((block[j].x >= volume.info.layers[layer_index].width_in_blocks)
                                            || (block[j].y >= volume.info.layers[layer_index].height_in_blocks)
                                            || (block[j].z >= volume.info.layers[layer_index].depth_in_blocks));

        if (ray_outside.isNotEmpty()) {
          slab_range[j].min(ray_outside) = range[j].max;
          slab_range[j].max(ray_outside) = range[j].max;

          mask[j] = mask[j] && !ray_outside;

          if (mask[j].isEmpty()) {
            continue;
          }
        }

        simd::float_v node_min;
        simd::float_v node_max;

        for (uint32_t k = 0; k < simd::len; k++) {
          if (mask[j][k]) {
            uint64_t node_handle = volume.info.node_handle(block[j].x[k], block[j].y[k], block[j].z[k], layer_index);
            node_min[k] = volume.nodes[node_handle].min;
            node_max[k] = volume.nodes[node_handle].max;
          }
        }

        simd::vec4 node_rgba = transfer_function(node_min, node_max, mask[j]);

        simd::float_m block_empty = mask[j] && (node_rgba.a == 0.f);

        // Empty space skipping
        if (block_empty.isNotEmpty()) {
          blend(transfer_function(slab_start_value[j], node_min, block_empty), dst[j], slab_range[j].max - slab_range[j].min, block_empty); // finish previous step with block value

          slab_range[j].min(block_empty) = range[j].max;
          slab_range[j].max(block_empty) = range[j].max;

          mask[j] = mask[j] && !block_empty;

          if (mask[j].isEmpty()) {
            continue;
          }
        }

        simd::float_m node_uniform = mask[j] && (node_min == node_max);

        // Fast integration of uniform space
        if (node_uniform.isNotEmpty()) {
          blend(transfer_function(slab_start_value[j], node_min, node_uniform), dst[j], slab_range[j].max - slab_range[j].min, node_uniform); // finish previous step with block value
          blend(node_rgba, dst[j], range[j].max - slab_range[j].max, node_uniform); // blend the rest of the block

          slab_start_value[j](node_uniform) = node_min;
          slab_range[j].min(node_uniform) = range[j].max;
          slab_range[j].max(node_uniform) = range[j].max + stepsize;

          mask[j] = mask[j] && !node_uniform;
        }
      }

      MaskPacklet integrate_mask = integrate_predicate(cell, layer, mask);

      for (uint32_t j = 0; j < simd::len; j++) {
        // Numeric integration
        for (integrate_mask[j] &= mask[j] & (slab_range[j].max < range[j].max); integrate_mask[j].isNotEmpty(); integrate_mask[j] &= slab_range[j].max < range[j].max) {
          simd::vec3 pos = ray[j].origin + ray[j].direction * slab_range[j].max;

          simd::vec3 in_block = (pos - cell[j]) * simd::float_v(exp2i(layer)) * simd::float_v(TreeVolume<T>::SUBVOLUME_SIDE);

          simd::float_v slab_end_value;

          for (uint32_t k = 0; k < simd::len; k++) {
            if (integrate_mask[j][k]) {
              uint64_t node_handle = volume.info.node_handle(block[j].x[k], block[j].y[k], block[j].z[k], layer_index);
              slab_end_value[k] = linterp(samplet(volume, volume.nodes[node_handle].block_handle, in_block.x[k], in_block.y[k], in_block.z[k]));
            }
          }

          blend(transfer_function(slab_start_value[j], slab_end_value, integrate_mask[j]), dst[j], slab_range[j].max - slab_range[j].min, integrate_mask[j]);

          slab_start_value[j](integrate_mask[j]) = slab_end_value;
          slab_range[j].min(integrate_mask[j]) = slab_range[j].max;
          slab_range[j].max(integrate_mask[j]) = slab_range[j].max + stepsize;
        }
      }
    });
  }

  return dst;
}
