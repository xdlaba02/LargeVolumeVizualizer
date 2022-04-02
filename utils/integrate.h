#pragma once

#include <ray/ray.h>
#include <ray/ray_simd.h>
#include <ray/intersection.h>
#include <ray/intersection_simd.h>

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
void integrate_simd(const simd::Ray &ray, float step, const F &func) {
  simd::RayRange range = intersect_aabb_ray(ray, {0, 0, 0}, { 1, 1, 1 });

  for (simd::float_m mask = range.min < range.max; mask.isNotEmpty(); mask &= range.min < range.max) {
    simd::vec3 sample_pos = ray.origin + ray.direction * range.min;

    func(sample_pos, mask);

    range.min += step;
  }
}
