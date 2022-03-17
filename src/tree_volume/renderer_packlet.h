#pragma once

#include "tree_volume.h"
#include "sampler.h"
#include "ray_octree_traversal_packlet.h"
#include "renderer_simd.h"
#include "../intersection.h"
#include "../utils.h"
#include "../simd.h"

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename T, typename TransferFunctionType>
Vec4Packlet render(const TreeVolume<T> &volume, const RayPacklet &ray_packlet, float step, MaskPacklet mask_packlet, const TransferFunctionType &transfer_function) {

  RayRangePacklet full_range_packlet;
  Vec4Packlet dst_packlet;
  RayRangePacklet slab_range_packlet;
  FloatPacklet slab_start_value_packlet;

  bool packlet_not_empty = false;
  for (uint32_t j = 0; j < simd::len; j++) {

    if (mask_packlet[j].isNotEmpty()) {
      intersect_aabb_ray(ray_packlet[j].origin, ray_packlet[j].direction_inverse, {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, full_range_packlet[j].min, full_range_packlet[j].max);

      mask_packlet[j] = mask_packlet[j] && full_range_packlet[j].min < full_range_packlet[j].max;
      dst_packlet[j] = {0.f, 0.f, 0.f, 1.f};

      if (mask_packlet[j].isNotEmpty()) {
        packlet_not_empty = true;
        slab_range_packlet[j] = { full_range_packlet[j].min, full_range_packlet[j].min };
      }
    }
  }

  if (packlet_not_empty) {
    auto ray_kernel = [&](const RayRangePacklet &range_packlet, const Vec3Packlet &cell_packlet, uint32_t layer, MaskPacklet &mask_packlet) {
      uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

      for (uint32_t j = 0; j < simd::len; j++) {
        if (mask_packlet[j].isEmpty()) {
          continue;
        }

        mask_packlet[j] = mask_packlet[j] && dst_packlet[j].a > 0.01f;

        if (mask_packlet[j].isEmpty()) {
          continue;
        }

        // Next step does not intersect the block
        mask_packlet[j] = mask_packlet[j] && slab_range_packlet[j].max < range_packlet[j].max;

        if (mask_packlet[j].isEmpty()) {
          continue;
        }

        Vec3Vec block = cell_packlet[j] * simd::float_v(approx_exp2(layer));

        simd::float_m ray_outside = mask_packlet[j] && ((block.x >= volume.info.layers[layer_index].width_in_blocks)
                                                    || (block.y >= volume.info.layers[layer_index].height_in_blocks)
                                                    || (block.z >= volume.info.layers[layer_index].depth_in_blocks));

        if (ray_outside.isNotEmpty()) {
          slab_range_packlet[j].min(ray_outside) = range_packlet[j].max;
          slab_range_packlet[j].max(ray_outside) = range_packlet[j].max;

          mask_packlet[j] = mask_packlet[j] && !ray_outside;

          if (mask_packlet[j].isEmpty()) {
            continue;
          }
        }

        float stepsize = step * approx_exp2(layer_index);

        std::array<uint64_t, simd::len> node_handle;
        simd::float_v node_min;
        simd::float_v node_max;

        for (uint32_t k = 0; k < simd::len; k++) {
          if (mask_packlet[j][k]) {
            node_handle[k] = volume.info.node_handle(block[0][k], block[1][k], block[2][k], layer_index);
            node_min[k] = volume.nodes[node_handle[k]].min;
            node_max[k] = volume.nodes[node_handle[k]].max;
          }
        }

        Vec4Vec node_rgba = transfer_function(node_min, node_max, mask_packlet[j]);

        simd::float_m block_empty = mask_packlet[j] && (node_rgba.a == 0.f);

        // Empty space skipping
        if (block_empty.isNotEmpty()) {
          blend(transfer_function(slab_start_value_packlet[j], node_min, block_empty), dst_packlet[j], slab_range_packlet[j].max - slab_range_packlet[j].min, block_empty); // finish previous step with block value

          slab_range_packlet[j].min(block_empty) = range_packlet[j].max;
          slab_range_packlet[j].max(block_empty) = range_packlet[j].max;

          mask_packlet[j] = mask_packlet[j] && !block_empty;

          if (mask_packlet[j].isEmpty()) {
            continue;
          }
        }

        simd::float_m node_uniform = mask_packlet[j] && (node_min == node_max);

        // Fast integration of uniform space
        if (node_uniform.isNotEmpty()) {
          blend(transfer_function(slab_start_value_packlet[j], node_min, node_uniform), dst_packlet[j], slab_range_packlet[j].max - slab_range_packlet[j].min, node_uniform); // finish previous step with block value
          blend(node_rgba, dst_packlet[j], range_packlet[j].max - slab_range_packlet[j].max, node_uniform); // blend the rest of the block

          slab_start_value_packlet[j](node_uniform) = node_min;
          slab_range_packlet[j].min(node_uniform) = range_packlet[j].max;
          slab_range_packlet[j].max(node_uniform) = range_packlet[j].max + stepsize;

          mask_packlet[j] = mask_packlet[j] && !node_uniform;

          if (mask_packlet[j].isEmpty()) {
            continue;
          }
        }

        // Recurse condition
        if (layer_index > 0) {
          continue;
        }

        // Numeric integration
        for (mask_packlet[j] = mask_packlet[j] && (slab_range_packlet[j].max < range_packlet[j].max); mask_packlet[j].isNotEmpty(); mask_packlet[j] = mask_packlet[j] && (slab_range_packlet[j].max < range_packlet[j].max)) {
          Vec3Vec pos = ray_packlet[j].origin + ray_packlet[j].direction * slab_range_packlet[j].max;

          Vec3Vec in_block = (pos - cell_packlet[j]) * simd::float_v(approx_exp2(layer)) * simd::float_v(TreeVolume<T>::SUBVOLUME_SIDE);

          simd::float_v slab_end_value;

          for (uint32_t k = 0; k < simd::len; k++) {
            if (mask_packlet[j][k]) {
              slab_end_value[k] = linterp(samplet(volume, volume.nodes[node_handle[k]].block_handle, in_block.x[k], in_block.y[k], in_block.z[k]));
            }
          }

          blend(transfer_function(slab_start_value_packlet[j], slab_end_value, mask_packlet[j]), dst_packlet[j], slab_range_packlet[j].max - slab_range_packlet[j].min, mask_packlet[j]);

          slab_start_value_packlet[j](mask_packlet[j]) = slab_end_value;
          slab_range_packlet[j].min(mask_packlet[j]) = slab_range_packlet[j].max;
          slab_range_packlet[j].max(mask_packlet[j]) = slab_range_packlet[j].max + stepsize;
        }
      }
    };

    ray_octree_traversal(ray_packlet, full_range_packlet, { }, 0, mask_packlet, ray_kernel);
  }

  return dst_packlet;
}
