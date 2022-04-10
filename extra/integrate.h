#pragma once

#include <components/ray/ray.h>
#include <components/ray/ray_simd.h>
#include <components/ray/intersection.h>
#include <components/ray/intersection_simd.h>
#include <components/ray/traversal_octree_packlet.h>

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
