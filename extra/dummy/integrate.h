#pragma once

#include <ray/ray.h>
#include <ray/ray_simd.h>
#include <ray/intersection.h>
#include <ray/intersection_simd.h>
#include <ray/traversal_octree.h>
#include <ray/traversal_octree_simd.h>
#include <ray/traversal_octree_packlet.h>

#include <tree_volume/tree_volume.h>

template <typename F>
void integrate_scalar(const Ray &ray, float step, const F &func) {
  RayRange range = intersect_aabb_ray(ray, {0, 0, 0}, { 1, 1, 1 });

  while (range.min < range.max) {
    glm::vec3 sample_pos = ray.origin + ray.direction * range.min;

    func(sample_pos);

    range.min += step;
  }
}

template <typename F>
void integrate_simd(const simd::Ray &ray, float step, simd::float_m mask, const F &func) {
  simd::RayRange range = intersect_aabb_ray(ray, {0, 0, 0}, { 1, 1, 1 });

  for (mask &= range.min < range.max; mask.isNotEmpty(); mask &= range.min < range.max) {
    simd::vec3 sample_pos = ray.origin + ray.direction * range.min;

    func(sample_pos, mask);

    range.min += step;
  }
}

template <typename F>
void integrate_packlet(const RayPacklet &ray, float step, MaskPacklet mask, const F &func) {
  RayRangePacklet range;

  for (uint8_t j = 0; j < simd::len; j++) {
    if (mask[j].isNotEmpty()) {
      range[j] = intersect_aabb_ray(ray[j], {0, 0, 0}, { 1, 1, 1 });
    }
  }

  auto update = [&] {
    bool isNotEmpty = false;
    for (uint8_t j = 0; j < simd::len; j++) {
      mask[j] &= range[j].min < range[j].max;
      if (mask[j].isNotEmpty()) {
        simd::vec3 sample_pos = ray[j].origin + ray[j].direction * range[j].min;
        func(sample_pos, mask[j]);
        range[j].min += step;
        isNotEmpty = true;
      }
    }
    return isNotEmpty;
  };

  while (update()) {}
}

template <typename T, uint32_t N, typename P, typename F>
void integrate_tree_scalar(const typename TreeVolume<T, N>::Info &info, const Ray &ray, float step, const P &integrate_predicate, const F &func) {
  RayRange global_range = intersect_aabb_ray(ray, {0.f, 0.f, 0.f}, { info.width_frac, info.height_frac, info.depth_frac });

  if (global_range.min < global_range.max) {

    ray_octree_traversal(ray, global_range, { 0.f, 0.f, 0.f }, 0, [&](const RayRange &range, const glm::vec3 &cell, uint32_t layer) {

      // Next step does not intersect the block
      if (global_range.min >= range.max) {
        return false;
      }

      uint8_t layer_index = std::size(info.layers) - 1 - layer;

      glm::vec3 node_pos = cell * exp2i(layer);

      if (node_pos.x >= info.layers[layer_index].width_in_nodes
      || (node_pos.y >= info.layers[layer_index].height_in_nodes)
      || (node_pos.z >= info.layers[layer_index].depth_in_nodes)) {
        global_range.min = range.max;
        return false;
      }

      float coef = exp2i(layer) * float(TreeVolume<T, N>::SUBVOLUME_SIDE);
      glm::vec3 offset = (ray.origin - cell) * coef;
      glm::vec3 mult = ray.direction * coef;

      if (integrate_predicate(cell, layer)) {
        // Numeric integration
        while (global_range.min < range.max) {
          glm::vec3 in_block_pos = offset + mult * global_range.min;

          func(node_pos, layer_index, in_block_pos);

          global_range.min += step;
        }

        return false;
      }

      return true;
    });
  }
}

template <typename T, uint32_t N, typename P, typename F>
void integrate_tree_simd(const typename TreeVolume<T, N>::Info &info, const simd::Ray &ray, float step, simd::float_m mask, const P &integrate_predicate, const F &func) {
  simd::RayRange global_range = intersect_aabb_ray(ray, {0.f, 0.f, 0.f}, { info.width_frac, info.height_frac, info.depth_frac});

  mask &= global_range.min < global_range.max;

  if (mask.isNotEmpty()) {
    ray_octree_traversal(ray, global_range, {}, 0, mask, [&](const simd::RayRange &range, const simd::vec3 &cell, uint32_t layer, simd::float_m &mask) {
      uint8_t layer_index = std::size(info.layers) - 1 - layer;

      // Next step does not intersect the block
      mask &= global_range.min < range.max;

      if (mask.isEmpty()) {
        return;
      }

      simd::vec3 node_pos = cell * simd::float_v(exp2i(layer));

      if (simd::float_m ray_outside = mask && ((node_pos.x >= info.layers[layer_index].width_in_nodes)
                                           || (node_pos.y >= info.layers[layer_index].height_in_nodes)
                                           || (node_pos.z >= info.layers[layer_index].depth_in_nodes)); ray_outside.isNotEmpty()) {
        global_range.min(ray_outside) = range.max;

        mask &= !ray_outside;

        if (mask.isEmpty()) {
          return;
        }
      }

      if (simd::float_m should_integrate = mask & integrate_predicate(cell, layer, mask); should_integrate.isNotEmpty()) {

        mask &= !should_integrate;

        simd::float_v coef = exp2i(layer) * float(TreeVolume<T, N>::SUBVOLUME_SIDE);

        simd::vec3 shift = (ray.origin - cell) * coef;
        simd::vec3 mult = ray.direction * coef;

        // Numeric integration
        for (should_integrate &= global_range.min < range.max; should_integrate.isNotEmpty(); should_integrate &= global_range.min < range.max) {
          simd::vec3 in_block_pos = global_range.min * mult + shift;

          func(node_pos, layer_index, in_block_pos, should_integrate);

          global_range.min(should_integrate) = global_range.min + step;
        }
      }
    });
  }
}

template <typename T, uint32_t N, typename P, typename F>
void integrate_tree_packlet(const typename TreeVolume<T, N>::Info &info, const RayPacklet &ray, float step, MaskPacklet mask, const P &integrate_predicate, const F &func) {
  RayRangePacklet global_range;

  bool packlet_not_empty = false;
  for (uint32_t j = 0; j < simd::len; j++) {

    if (mask[j].isNotEmpty()) {
      global_range[j] = intersect_aabb_ray(ray[j], {0.f, 0.f, 0.f}, { info.width_frac, info.height_frac, info.depth_frac});

      mask[j] &= global_range[j].min < global_range[j].max;

      packlet_not_empty |= mask[j].isNotEmpty();
    }
  }

  if (packlet_not_empty) {
    ray_octree_traversal(ray, global_range, { }, 0, mask, [&](const RayRangePacklet &range, const Vec3Packlet &cell, uint8_t layer, MaskPacklet &mask) {
      uint8_t layer_index = std::size(info.layers) - 1 - layer;

      Vec3Packlet node_pos;

      for (uint32_t j = 0; j < simd::len; j++) {
        if (mask[j].isEmpty()) {
          continue;
        }

        // Next step does not intersect the block
        mask[j] &= global_range[j].min < range[j].max;

        if (mask[j].isEmpty()) {
          continue;
        }

        node_pos[j] = cell[j] * simd::float_v(exp2i(layer));

        if (simd::float_m ray_outside = mask[j] && ((node_pos[j].x >= info.layers[layer_index].width_in_nodes)
                                                || (node_pos[j].y >= info.layers[layer_index].height_in_nodes)
                                                || (node_pos[j].z >= info.layers[layer_index].depth_in_nodes)); ray_outside.isNotEmpty()) {

          global_range[j].min(ray_outside) = range[j].max;

          mask[j] &= !ray_outside;

          if (mask[j].isEmpty()) {
            continue;
          }
        }
      }

      MaskPacklet should_integrate = integrate_predicate(cell, layer, mask);

      simd::float_v coef = exp2i(layer) * float(TreeVolume<T, N>::SUBVOLUME_SIDE);

      for (uint32_t j = 0; j < simd::len; j++) {

        should_integrate[j] &= mask[j];

        mask[j] &= !should_integrate[j];

        simd::vec3 shift = (ray[j].origin - cell[j]) * coef;
        simd::vec3 mult = ray[j].direction * coef;

        for (should_integrate[j] &= (global_range[j].min < range[j].max); should_integrate[j].isNotEmpty(); should_integrate[j] &= global_range[j].min < range[j].max) {
          simd::vec3 in_block_pos = global_range[j].min * mult + shift;

          func(node_pos[j], layer_index, in_block_pos, should_integrate[j]);

          global_range[j].min(should_integrate[j]) += step;
        }
      }
    });
  }
}
