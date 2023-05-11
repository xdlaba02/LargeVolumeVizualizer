/**
* @file texture2D.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Class representing generic, simple and slow 2D texture.
*/

#pragma once

#include <cstdint>

#include <vector>

template <typename T>
class Texture2D {
public:
  Texture2D(uint32_t width, uint32_t height)
      : m_width(width)
      , m_height(height)
      , m_data(width * height) {
  }

  uint32_t width() const { return m_width; }
  uint32_t height() const { return m_height; }

  T &operator()(uint32_t x, uint32_t y) {
    return m_data[y * m_width + x];
  }

  const T &operator()(uint32_t x, uint32_t y) const {
    return m_data[y * m_width + x];
  }

private:
  uint32_t m_width;
  uint32_t m_height;
  std::vector<T> m_data;
};
