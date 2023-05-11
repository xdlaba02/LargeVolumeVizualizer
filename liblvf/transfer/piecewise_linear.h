/**
* @file piecewise_linear.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Function that samples function defined by keys and values inside a map via linear interpolation.
*/

#pragma once

#include <map>
#include <cassert>

template <typename T>
T piecewise_linear(const std::map<float, T> &values, float value) {
  assert(std::size(values));

  auto it = values.lower_bound(value);

  if (it == std::end(values)) {
    return std::rbegin(values)->second;
  }
  else if (it == std::begin(values)){
    return it->second;
  }
  else {
    auto lowerIt = std::prev(it);
    return lowerIt->second + (it->second - lowerIt->second) * (value - lowerIt->first) / (it->first - lowerIt->first);
  }
}
