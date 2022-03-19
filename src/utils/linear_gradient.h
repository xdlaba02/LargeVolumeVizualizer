#pragma once

#include <map>
#include <cassert>

template <typename ColorType>
ColorType linear_gradient(const std::map<float, ColorType> &colors, float value) {
  assert(std::size(colors));

  auto it = colors.lower_bound(value);

  if (it == std::end(colors)) {
    return std::rbegin(colors)->second;
  }
  else if (it == std::begin(colors)){
    return it->second;
  }
  else {
    auto lowerIt = std::prev(it);
    return lowerIt->second + (it->second - lowerIt->second) * (value - lowerIt->first) / (it->first - lowerIt->first);
  }
}
