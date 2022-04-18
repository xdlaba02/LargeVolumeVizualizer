#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler_simd.h>

#include <ray/traversal_octree_packlet.h>
#include <ray/intersection.h>

#include <utils/fast_exp2.h>
#include <utils/simd.h>
#include <utils/blend_simd.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename T, uint32_t N, typename F, typename P>
Vec4Packlet integrate_tree_slab_packlet(const TreeVolume<T, N> &volume, const RayPacklet &ray, float step, float terminate_thresh, MaskPacklet mask, const F &transfer_function, const P &integrate_predicate) {
  Vec4Packlet dst;

  using BlockHandlePacklet = std::array<std::array<uint64_t, simd::len>, simd::len>;

  RayRangePacklet range;
  RayRangePacklet slab_range;
  FloatPacklet slab_start_value;
  BlockHandlePacklet block_handle;

  bool packlet_not_empty = false;
  for (uint32_t j = 0; j < simd::len; j++) {

    if (mask[j].isNotEmpty()) {
      range[j] = intersect_aabb_ray(ray[j], {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac});

      mask[j] &= range[j].min < range[j].max;
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

      Vec3Packlet node_pos;

      for (uint32_t j = 0; j < simd::len; j++) {
        if (mask[j].isEmpty()) {
          continue;
        }

        mask[j] &= dst[j].a > terminate_thresh;

        if (mask[j].isEmpty()) {
          continue;
        }

        // Next step does not intersect the block
        mask[j] &= slab_range[j].max < range[j].max;

        if (mask[j].isEmpty()) {
          continue;
        }

        node_pos[j] = cell[j] * simd::float_v(exp2i(layer));

        if (simd::float_m ray_outside = mask[j] && ((node_pos[j].x >= volume.info.layers[layer_index].width_in_nodes)
                                                || (node_pos[j].y >= volume.info.layers[layer_index].height_in_nodes)
                                                || (node_pos[j].z >= volume.info.layers[layer_index].depth_in_nodes)); ray_outside.isNotEmpty()) {
          slab_range[j].min(ray_outside) = range[j].max;
          slab_range[j].max(ray_outside) = range[j].max;

          mask[j] &= !ray_outside;

          if (mask[j].isEmpty()) {
            continue;
          }
        }

        simd::float_v node_min;
        simd::float_v node_max;

        for (uint32_t k = 0; k < simd::len; k++) {
          if (mask[j][k]) {
            uint64_t node_handle = volume.info.node_handle(node_pos[j].x[k], node_pos[j].y[k], node_pos[j].z[k], layer_index);
            node_min[k] = volume.node(node_handle).min;
            node_max[k] = volume.node(node_handle).max;
            block_handle[j][k] = volume.node(node_handle).block_handle;
          }
        }

        simd::vec4 node_rgba = transfer_function(node_min, node_max, mask[j]);

        // Empty space skipping
        if (simd::float_m block_empty = mask[j] && (node_rgba.a == 0.f); block_empty.isNotEmpty()) {
          blend(transfer_function(slab_start_value[j], node_min, block_empty), dst[j], slab_range[j].max - slab_range[j].min, block_empty); // finish previous step with block value

          slab_range[j].min(block_empty) = range[j].max;
          slab_range[j].max(block_empty) = range[j].max;

          mask[j] &= !block_empty;

          if (mask[j].isEmpty()) {
            continue;
          }
        }

        // Fast integration of uniform space
        if (simd::float_m node_uniform = mask[j] && (node_min == node_max); node_uniform.isNotEmpty()) {
          blend(transfer_function(slab_start_value[j], node_min, node_uniform), dst[j], slab_range[j].max - slab_range[j].min, node_uniform); // finish previous step with block value
          blend(node_rgba, dst[j], range[j].max - slab_range[j].max, node_uniform); // blend the rest of the block

          slab_start_value[j](node_uniform) = node_min;
          slab_range[j].min(node_uniform) = range[j].max;
          slab_range[j].max(node_uniform) = range[j].max + step;

          mask[j] &= !node_uniform;
        }
      }

      MaskPacklet should_integrate = integrate_predicate(dst, layer, mask);

      simd::float_v coef = exp2i(layer) * float(TreeVolume<T, N>::SUBVOLUME_SIDE);

      for (uint32_t j = 0; j < simd::len; j++) {

        should_integrate[j] &= mask[j];
        mask[j] &= !should_integrate[j];

        simd::vec3 shift = (ray[j].origin - cell[j]) * coef;
        simd::vec3 mult = ray[j].direction * coef;

        for (should_integrate[j] &= (slab_range[j].max < range[j].max); should_integrate[j].isNotEmpty(); should_integrate[j] &= slab_range[j].max < range[j].max && dst[j].a > terminate_thresh) {
          simd::vec3 in_block_pos = slab_range[j].max * mult + shift;

          simd::float_v slab_end_value = sample(volume, block_handle[j], in_block_pos.x, in_block_pos.y, in_block_pos.z, should_integrate[j]);

          blend(transfer_function(slab_start_value[j], slab_end_value, should_integrate[j]), dst[j], slab_range[j].max - slab_range[j].min, should_integrate[j]);

          slab_start_value[j](should_integrate[j]) = slab_end_value;
          slab_range[j].min(should_integrate[j]) = slab_range[j].max;
          slab_range[j].max(should_integrate[j]) = slab_range[j].max + step;
        }
      }
    });
  }

  return dst;
}
