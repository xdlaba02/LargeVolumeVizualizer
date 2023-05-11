/**
* @file ray_generator.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Class that generates persepective rays for pixels on screen.
* The rays are in view space and can be further altered by generic transform matrices.
*/

#pragma once

#include <glm/glm.hpp>

#include <cstdint>

class RayGenerator {
public:
  RayGenerator(uint32_t width, uint32_t height, float yfov_degrees) {
    float yfov = glm::radians(yfov_degrees);

    float yfov_coef = std::tan(yfov / 2.f);
    float xfov_coef = yfov_coef * width / height;

    float width_coef_avg  = xfov_coef / width;
    m_width_coef  = 2.f * width_coef_avg;
    m_width_shift = width_coef_avg - xfov_coef;

    float height_coef_avg = yfov_coef / height;
    m_height_coef = 2.f * height_coef_avg;
    m_height_shift = height_coef_avg - yfov_coef;
  }

  inline glm::vec4 operator()(uint32_t x, uint32_t y) const {
    return glm::normalize(glm::vec4(x * m_width_coef + m_width_shift, y * m_height_coef + m_height_shift, -1.f, 0.f));
  }

private:
  float m_width_coef;
  float m_width_shift;

  float m_height_coef;
  float m_height_shift;
};
