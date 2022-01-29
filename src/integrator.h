#pragma once

#include "blocked_volume.h"
#include "preintegrated_transfer_function.h"
#include "intersection.h"

#include <glm/glm.hpp>

#include <cstdint>

template <typename F>
class Integrator {
public:
  Integrator(const F &transfer):
      m_transfer_r([&](float v){ return transfer(v).r; }),
      m_transfer_g([&](float v){ return transfer(v).g; }),
      m_transfer_b([&](float v){ return transfer(v).b; }),
      m_transfer_a([&](float v){ return transfer(v).a; }) {}

  glm::vec4 integrate(const BlockedVolume<uint8_t> &volume, const glm::vec3 &origin, const glm::vec3 &direction) {
    float tmin, tmax;
    intersect_aabb_ray(origin, direction, {0, 0, 0}, {volume.width() - 1, volume.height() - 1, volume.depth() - 1}, tmin, tmax);

    glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

    glm::vec3 sample = origin + direction * tmin;
    glm::vec3 step = direction * m_stepsize;

    float prev_value {};

    if (tmin <= tmax) {
      prev_value = volume.sample(sample.x, sample.y, sample.z);

      tmin += m_stepsize;
      sample += step;
    }

    while (tmin <= tmax) {

      float value = volume.sample(sample.x, sample.y, sample.z);

      float a = m_transfer_a(value, prev_value);

      if (a > 0.f) {
        float r = m_transfer_r(value, prev_value);
        float g = m_transfer_g(value, prev_value);
        float b = m_transfer_b(value, prev_value);

        float alpha = 1.f - std::exp(-a * std::min(m_stepsize, tmax - tmin));

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

      tmin += m_stepsize;
      sample += step;
    }

    return dst;
  }

  glm::vec<4, simd::float_v> integrate(const BlockedVolume<uint8_t> &volume, const glm::vec3 &origin, const glm::vec<3, simd::float_v> &directions) {
    simd::float_v tmins, tmaxs;
    intersect_aabb_rays_single_origin(origin, directions, {0, 0, 0}, {volume.width() - 1, volume.height() - 1, volume.depth() - 1}, tmins, tmaxs);

    glm::vec<4, simd::float_v> dsts(0.f, 0.f, 0.f, 1.f);

    glm::vec<3, simd::float_v> samples = glm::vec<3, simd::float_v>(origin) + directions * tmins;
    glm::vec<3, simd::float_v> steps = directions * simd::float_v(m_stepsize);

    simd::float_v prev_values = volume.samples(samples.x, samples.y, samples.z, tmins <= tmaxs);

    tmins += m_stepsize;
    samples += steps;

    for (simd::float_m mask = tmins <= tmaxs; !mask.isEmpty(); mask &= tmins <= tmaxs) {
      simd::float_v values = volume.samples(samples.x, samples.y, samples.z, mask);

      simd::float_v a = m_transfer_a(values, prev_values, mask);

      simd::float_m alpha_mask = a > 0.f;

      if (!alpha_mask.isEmpty()) {
        simd::float_v r = m_transfer_r(values, prev_values, mask & alpha_mask);
        simd::float_v g = m_transfer_g(values, prev_values, mask & alpha_mask);
        simd::float_v b = m_transfer_b(values, prev_values, mask & alpha_mask);

        simd::float_v alpha = simd::float_v(1.f) - Vc::exp(-a * Vc::min(m_stepsize, tmaxs - tmins));

        simd::float_v coef = alpha * dsts.a;

        dsts.r(mask & alpha_mask) += r * coef;
        dsts.g(mask & alpha_mask) += g * coef;
        dsts.b(mask & alpha_mask) += b * coef;
        dsts.a(mask & alpha_mask) *= 1 - alpha;

        mask &= dsts.a > 1.f / 256.f;
      }

      prev_values = values;

      tmins += m_stepsize;
      samples += steps;
    }

    return dsts;
  }

private:
  static constexpr float m_stepsize = 0.005f;

  PreintegratedTransferFunction<uint8_t> m_transfer_r;
  PreintegratedTransferFunction<uint8_t> m_transfer_g;
  PreintegratedTransferFunction<uint8_t> m_transfer_b;
  PreintegratedTransferFunction<uint8_t> m_transfer_a;
};
