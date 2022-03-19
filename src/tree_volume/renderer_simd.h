#pragma once

#include "tree_volume.h"
#include "sampler.h"

#include <ray_traversal/octree_traversal_simd.h>
#include <ray_traversal/intersection_simd.h>

#include <utils/utils.h>
#include <utils/simd.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

void blend(const Vec4Vec &src_vec, Vec4Vec &dst_vec, simd::float_v stepsize, const simd::float_m &mask_vec) {
  simd::float_v alpha = 1.f - simd::exp(-src_vec.a * stepsize);

  simd::float_v coef = alpha * dst_vec.a;

  dst_vec.r(mask_vec) += src_vec.r * coef;
  dst_vec.g(mask_vec) += src_vec.g * coef;
  dst_vec.b(mask_vec) += src_vec.b * coef;
  dst_vec.a(mask_vec) *= 1.f - alpha;
};

template <typename T, typename TransferFunctionType>
Vec4Vec render(const TreeVolume<T> &volume, const RayVec &ray_vec, float step, simd::float_m mask_vec, const TransferFunctionType &transfer_function) {

  Vec4Vec dst_vec = {0.f, 0.f, 0.f, 1.f};
  RayRangeVec full_range_vec;

  intersect_aabb_ray(ray_vec.origin, ray_vec.direction_inverse, {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, full_range_vec.min, full_range_vec.max);

  mask_vec = mask_vec && full_range_vec.min < full_range_vec.max;

  if (mask_vec.isNotEmpty()) {
    simd::float_v slab_start_value_vec;
    RayRangeVec slab_range_vec = { full_range_vec.min, full_range_vec.min };

    auto ray_kernel = [&](const RayRangeVec &range_vec, const Vec3Vec &cell_vec, uint32_t layer, simd::float_m &mask_vec) {
      uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

      mask_vec = mask_vec && dst_vec.a > 0.01f;

      if (mask_vec.isEmpty()) {
        return;
      }

      // Next step does not intersect the block
      mask_vec = mask_vec && slab_range_vec.max < range_vec.max;

      if (mask_vec.isEmpty()) {
        return;
      }

      Vec3Vec block = cell_vec * simd::float_v(approx_exp2(layer));

      simd::float_m ray_outside = mask_vec && ((block.x >= volume.info.layers[layer_index].width_in_blocks)
                                           || (block.y >= volume.info.layers[layer_index].height_in_blocks)
                                           || (block.z >= volume.info.layers[layer_index].depth_in_blocks));

      if (ray_outside.isNotEmpty()) {
        slab_range_vec.min(ray_outside) = range_vec.max;
        slab_range_vec.max(ray_outside) = range_vec.max;

        mask_vec = mask_vec && !ray_outside;

        if (mask_vec.isEmpty()) {
          return;
        }
      }

      float stepsize = step * approx_exp2(layer_index);

      std::array<uint64_t, simd::len> node_handle;
      simd::float_v node_min;
      simd::float_v node_max;

      for (uint32_t k = 0; k < simd::len; k++) {
        if (mask_vec[k]) {
          node_handle[k] = volume.info.node_handle(block[0][k], block[1][k], block[2][k], layer_index);
          node_min[k] = volume.nodes[node_handle[k]].min;
          node_max[k] = volume.nodes[node_handle[k]].max;
        }
      }

      Vec4Vec node_rgba = transfer_function(node_min, node_max, mask_vec);

      simd::float_m block_empty = mask_vec && (node_rgba.a == 0.f);

      // Empty space skipping
      if (block_empty.isNotEmpty()) {
        blend(transfer_function(slab_start_value_vec, node_min, block_empty), dst_vec, slab_range_vec.max - slab_range_vec.min, block_empty); // finish previous step with block value

        slab_range_vec.min(block_empty) = range_vec.max;
        slab_range_vec.max(block_empty) = range_vec.max;

        mask_vec = mask_vec && !block_empty;

        if (mask_vec.isEmpty()) {
          return;
        }
      }

      simd::float_m node_uniform = mask_vec && (node_min == node_max);

      // Fast integration of uniform space
      if (node_uniform.isNotEmpty()) {
        blend(transfer_function(slab_start_value_vec, node_min, node_uniform), dst_vec, slab_range_vec.max - slab_range_vec.min, node_uniform); // finish previous step with block value
        blend(node_rgba, dst_vec, range_vec.max - slab_range_vec.max, node_uniform); // blend the rest of the block

        slab_start_value_vec(node_uniform) = node_min;
        slab_range_vec.min(node_uniform) = range_vec.max;
        slab_range_vec.max(node_uniform) = range_vec.max + stepsize;

        mask_vec = mask_vec && !node_uniform;

        if (mask_vec.isEmpty()) {
          return;
        }
      }

      // Recurse condition
      if (layer_index > 0) {
        return;
      }

      // Numeric integration
      for (mask_vec = mask_vec && (slab_range_vec.max < range_vec.max); mask_vec.isNotEmpty(); mask_vec = mask_vec && (slab_range_vec.max < range_vec.max)) {
        Vec3Vec pos = ray_vec.origin + ray_vec.direction * slab_range_vec.max;

        Vec3Vec in_block = (pos - cell_vec) * simd::float_v(approx_exp2(layer)) * simd::float_v(TreeVolume<T>::SUBVOLUME_SIDE);

        simd::float_v slab_end_value;

        for (uint32_t k = 0; k < simd::len; k++) {
          if (mask_vec[k]) {
            slab_end_value[k] = linterp(samplet(volume, volume.nodes[node_handle[k]].block_handle, in_block.x[k], in_block.y[k], in_block.z[k]));
          }
        }

        blend(transfer_function(slab_start_value_vec, slab_end_value, mask_vec), dst_vec, slab_range_vec.max - slab_range_vec.min, mask_vec);

        slab_start_value_vec(mask_vec) = slab_end_value;
        slab_range_vec.min(mask_vec) = slab_range_vec.max;
        slab_range_vec.max(mask_vec) = slab_range_vec.max + stepsize;
      }
    };

    ray_octree_traversal(ray_vec, full_range_vec, { }, 0, mask_vec, ray_kernel);
  }

  return dst_vec;
}
