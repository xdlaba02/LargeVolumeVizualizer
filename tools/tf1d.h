/**
* @file tf1d.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Function for loading transfer function files.
*/

#pragma once

#include <glm/glm.hpp>

#include <map>
#include <fstream>

struct TF1D {
  static TF1D load_from_file(const char *filename)
  {
    TF1D tf {};

    std::ifstream file(filename);

    if (!file) {
      throw std::runtime_error(std::string("Unable to open '") + filename + "'!");
    }

    uint32_t num_values;

    file >> num_values;

    for (uint32_t i = 0; i < num_values; i++) {
      float key {};
      glm::vec3 rgb {};
      file >> key >> rgb.r >> rgb.g >> rgb.b;
      tf.rgb[key] = rgb;
    }

    file >> num_values;

    for (uint32_t i = 0; i < num_values; i++) {
      float key {};
      float value {};
      file >> key >> value;
      tf.a[key] = value;
    }

    return tf;
  }

  std::map<float, glm::vec3> rgb;
  std::map<float, float> a;
};
