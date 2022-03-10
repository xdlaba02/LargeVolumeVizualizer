#pragma once

#include "tree_volume/tree_volume.h"
#include "preintegrated_transfer_function.h"
#include "intersection.h"

#include <glm/glm.hpp>

#include <cstdint>

template <typename T>
class Integrator {
public:
  template <typename F>
  Integrator(const F &transfer, float stepsize):
      m_transfer_r([&](float v){ return transfer(v).r; }),
      m_transfer_g([&](float v){ return transfer(v).g; }),
      m_transfer_b([&](float v){ return transfer(v).b; }),
      m_transfer_a([&](float v){ return transfer(v).a; }),
      m_stepsize(stepsize) {}

  glm::vec4 integrate(const TreeVolume<T> &volume, const glm::vec3 &origin, const glm::vec3 &direction, uint8_t layer) const {
    float tmin, tmax;
    intersect_aabb_ray(origin, 1.f / direction, {0, 0, 0}, {1, 1, 1}, tmin, tmax);

    float stepsize = m_stepsize * (layer + 1);

    glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

    float prev_value {};

    if (tmin < tmax) {
      glm::vec3 sample = origin + direction * tmin;
      prev_value = volume.sample_volume(sample.x, sample.y, sample.z, layer);
    }

    tmin += stepsize;

    while (tmin < tmax) {
      glm::vec3 sample = origin + direction * tmin;

      float value = volume.sample_volume(sample.x, sample.y, sample.z, layer);

      float a = m_transfer_a(value, prev_value);

      if (a > 0.f) {
        float r = m_transfer_r(value, prev_value);
        float g = m_transfer_g(value, prev_value);
        float b = m_transfer_b(value, prev_value);

        float alpha = 1.f - std::exp(-a * stepsize);

        float coef = alpha * dst.a;

        dst.r += r * coef;
        dst.g += g * coef;
        dst.b += b * coef;
        dst.a *= 1 - alpha;

        if (dst.a <= 1.f / 256.f) {
          break;
        }
      }

      prev_value = value;
      tmin += stepsize;
    }

    return dst;
  }

  glm::vec<4, simd::float_v> integrate(const TreeVolume<T> &volume, const glm::vec3 &origin, const glm::vec<3, simd::float_v> &directions, const simd::uint32_v &layer) const {
    simd::float_v tmins, tmaxs;
    intersect_aabb_rays_single_origin(origin, simd::float_v(1.f) / directions, {0, 0, 0}, {1, 1, 1}, tmins, tmaxs);

    simd::float_v stepsize = m_stepsize * simd::float_v(layer + 1);

    glm::vec<4, simd::float_v> dsts(0.f, 0.f, 0.f, 1.f);

    simd::float_v prev_values {};

    simd::float_m mask = tmins < tmaxs;

    if (!mask.isEmpty()) {
      glm::vec<3, simd::float_v> samples = glm::vec<3, simd::float_v>(origin) + directions * tmins;
      prev_values = volume.sample_volume(samples.x, samples.y, samples.z, layer, tmins < tmaxs);
    }

    tmins += stepsize;
    mask &= tmins < tmaxs;

    while (!mask.isEmpty()) {
      glm::vec<3, simd::float_v> samples = glm::vec<3, simd::float_v>(origin) + directions * tmins;

      simd::float_v values = volume.sample_volume(samples.x, samples.y, samples.z, layer, mask);

      simd::float_v a = m_transfer_a(values, prev_values, mask);

      simd::float_m alpha_mask = a > 0.f;

      if (!alpha_mask.isEmpty()) {
        simd::float_v r = m_transfer_r(values, prev_values, mask & alpha_mask);
        simd::float_v g = m_transfer_g(values, prev_values, mask & alpha_mask);
        simd::float_v b = m_transfer_b(values, prev_values, mask & alpha_mask);

        simd::float_v alpha = simd::float_v(1.f) - Vc::exp(-a * stepsize);

        simd::float_v coef = alpha * dsts.a;

        dsts.r(mask & alpha_mask) += r * coef;
        dsts.g(mask & alpha_mask) += g * coef;
        dsts.b(mask & alpha_mask) += b * coef;
        dsts.a(mask & alpha_mask) *= 1 - alpha;

        mask &= dsts.a > 1.f / 256.f;
      }

      prev_values = values;
      tmins += stepsize;
      mask &= tmins < tmaxs;
    }

    return dsts;
  }

  /*
  glm::vec4 integrate2(const TreeVolume<T> &volume, const glm::vec3 &origin, const glm::vec3 &direction) const {

    glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

    const glm::vec3 size = {volume.info.layers[0].width, volume.info.layers[0].height, volume.info.layers[0].depth};
    const glm::vec3 size_in_blocks = {volume.info.layers[0].width_in_blocks, volume.info.layers[0].height_in_blocks, volume.info.layers[0].depth_in_blocks};

    raster_traversal<TreeVolume<T>::SUBVOLUME_SIDE>(origin, direction, size, size_in_blocks, [&](const glm::vec<3, uint32_t> &block_pos, const glm::vec3 &in_block_pos, float tmax) {

      typename TreeVolume<T>::Node node = volume.nodes[volume.info.node_handle(block_pos.x, block_pos.y, block_pos.z, 0)];

      if (node.min == node.max) { // fast integration
        float a = m_transfer_a(node.min, node.max);

        if (a > 0.f) { // empty space skipping
          float r = m_transfer_r(node.min, node.max);
          float g = m_transfer_g(node.min, node.max);
          float b = m_transfer_b(node.min, node.max);

          float alpha = 1.f - std::exp(-a * tmax);

          float coef = alpha * dst.a;

          dst.r += r * coef;
          dst.g += g * coef;
          dst.b += b * coef;
          dst.a *= 1 - alpha;
        }
      }
      else { // integration by sampling
        float prev_value = volume.sample_block(node.block_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);

        float tmin = 0.f;

        while (tmin < tmax) {
          float next_step = std::min(m_stepsize, tmax - tmin);

          tmin += next_step;

          glm::vec3 current_pos = in_block_pos + direction * tmin;

          float value = volume.sample_block(node.block_handle, current_pos.x, current_pos.y, current_pos.z);

          float a = m_transfer_a(value, prev_value);

          if (a > 0.f) { // empty sample skipping
            float r = m_transfer_r(value, prev_value);
            float g = m_transfer_g(value, prev_value);
            float b = m_transfer_b(value, prev_value);

            float alpha = 1.f - std::exp(-a * next_step);

            float coef = alpha * dst.a;

            dst.r += r * coef;
            dst.g += g * coef;
            dst.b += b * coef;
            dst.a *= 1 - alpha;
          }

          prev_value = value;
        }
      }

      return dst.a > 1.f / 256.f;
    });

    return dst;
  }
  */
private:
  PreintegratedTransferFunction<T> m_transfer_r;
  PreintegratedTransferFunction<T> m_transfer_g;
  PreintegratedTransferFunction<T> m_transfer_b;
  PreintegratedTransferFunction<T> m_transfer_a;

  float m_stepsize;
};
