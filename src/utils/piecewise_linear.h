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
