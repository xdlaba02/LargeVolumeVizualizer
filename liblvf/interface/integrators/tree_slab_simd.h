#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler_simd.h>

#include <ray/traversal_octree_simd.h>
#include <ray/intersection_simd.h>

#include <utils/fast_exp2.h>
#include <utils/simd.h>
#include <utils/blend_simd.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>
#include <concepts>

template <typename T, uint32_t N, typename F, typename P>
simd::vec4 integrate_tree_slab_simd(const TreeVolume<T, N> &volume, const simd::Ray &ray, float step, float terminate_thresh, simd::float_m mask, const F &transfer_function, const P &integrate_predicate) {
  simd::vec4 dst = {0.f, 0.f, 0.f, 1.f};

  simd::RayRange range = intersect_aabb_ray(ray, {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac});

  mask &= range.min < range.max;

  if (mask.isNotEmpty()) {
    simd::RayRange slab_range { range.min, range.min };
    simd::float_v slab_start_value;

    ray_octree_traversal(ray, range, {}, 0, mask, [&](const simd::RayRange &range, const simd::vec3 &cell, uint32_t layer, simd::float_m &mask) {
      uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

      mask &= dst.a > terminate_thresh;

      if (mask.isEmpty()) {
        return;
      }

      // Next step does not intersect the block
      mask &= slab_range.max < range.max;

      if (mask.isEmpty()) {
        return;
      }

      simd::vec3 node_pos = cell * simd::float_v(exp2i(layer));

      if (simd::float_m ray_outside = mask && ((node_pos.x >= volume.info.layers[layer_index].width_in_nodes)
                                           ||  (node_pos.y >= volume.info.layers[layer_index].height_in_nodes)
                                           ||  (node_pos.z >= volume.info.layers[layer_index].depth_in_nodes)); ray_outside.isNotEmpty()) {
        slab_range.min(ray_outside) = range.max;
        slab_range.max(ray_outside) = range.max;

        mask &= !ray_outside;

        if (mask.isEmpty()) {
          return;
        }
      }

      std::array<uint64_t, simd::len> block_handle;
      simd::float_v node_min;
      simd::float_v node_max;

      for (uint32_t k = 0; k < simd::len; k++) {
        if (mask[k]) {
          uint64_t node_handle = volume.info.node_handle(node_pos[0][k], node_pos[1][k], node_pos[2][k], layer_index);
          node_min[k] = volume.node(node_handle).min;
          node_max[k] = volume.node(node_handle).max;
          block_handle[k] = volume.node(node_handle).block_handle;
        }
      }

      simd::vec4 node_rgba = transfer_function(node_min, node_max, mask);

      // Empty space skipping
      if (simd::float_m block_empty = mask & (node_rgba.a == 0.f); block_empty.isNotEmpty()) {
        blend(transfer_function(slab_start_value, node_min, block_empty), dst, slab_range.max - slab_range.min, block_empty); // finish previous step with block value

        slab_range.min(block_empty) = range.max;
        slab_range.max(block_empty) = range.max;

        mask &= !block_empty;

        if (mask.isEmpty()) {
          return;
        }
      }

      // Fast integration of uniform space
      if (simd::float_m node_uniform = mask & (node_min == node_max); node_uniform.isNotEmpty()) {
        blend(transfer_function(slab_start_value, node_min, node_uniform), dst, slab_range.max - slab_range.min, node_uniform); // finish previous step with block value
        blend(node_rgba, dst, range.max - slab_range.max, node_uniform); // blend the rest of the block

        slab_start_value(node_uniform) = node_min;
        slab_range.min(node_uniform) = range.max;
        slab_range.max(node_uniform) = range.max + step;

        mask &= !node_uniform;

        if (mask.isEmpty()) {
          return;
        }
      }

      // Numeric integration
      if (simd::float_m should_integrate = mask & integrate_predicate(dst.a, layer, mask); should_integrate.isNotEmpty()) {

        mask &= !should_integrate;

        simd::float_v coef = exp2i(layer) * float(TreeVolume<T, N>::SUBVOLUME_SIDE);

        simd::vec3 shift = (ray.origin - cell) * coef;
        simd::vec3 mult = ray.direction * coef;

        for (should_integrate &= slab_range.max < range.max; should_integrate.isNotEmpty(); should_integrate &= slab_range.max < range.max && dst.a > terminate_thresh) {
          simd::vec3 in_block_pos = slab_range.max * mult + shift;

          simd::float_v slab_end_value = sample(volume, block_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z, should_integrate);

          blend(transfer_function(slab_start_value, slab_end_value, should_integrate), dst, slab_range.max - slab_range.min, should_integrate);

          slab_start_value(should_integrate) = slab_end_value;
          slab_range.min(should_integrate) = slab_range.max;
          slab_range.max(should_integrate) = slab_range.max + step;
        }
      }
    });
  }

  return dst;
}
