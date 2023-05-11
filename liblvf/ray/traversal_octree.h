/**
* @file traversal_octree.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Generic function for traversing an octree with a ray.
* This is a piece of art because it works the same way as integer arithmetic variant but without conversion to float.
*/

#pragma once

#include "ray.h"

#include <utils/fast_exp2.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename F>
void ray_octree_traversal(const Ray &ray, const RayRange &range, glm::vec3 cell, uint32_t layer, const F &callback) {
  // This returns current state of the traversal via callback.
  // The user can do whatever he wants with it (check corresponding octree node, integrate it) and decide if the wants to recurse.
  // This serves as a recursion end condition.
  if (callback(range, cell, layer)) {

    // This computes a side of a child node.
    // For layer 0, this computes 2^(-1), which is exacly 0.5f.
    // For layer 1, child node is 2^(-2), which is 0.25f, etc.
    // This numbers can be represented exactly by float and generated just by changing exponent of the 1.f value, which is what exp2i() does.
    // This is fixed point arithmetic trick implemented into floating point because now I can use the number with geometry operation without conversions.
    // This shall be precomputed in future version without recursion.
    float child_size = exp2i(-layer - 1);

    // This computes a center of a current node.
    glm::vec3 center = cell + child_size;

    // This performs intersection of ray with three axis aligned planes defined by center point.
    glm::vec3 tcenter = (center - ray.origin) * ray.direction_inverse;

    // Fast indirect sort of tcenter into axis.
    std::array<uint8_t, 3> axis;

    bool gt01 = tcenter[0] > tcenter[1];
    bool gt02 = tcenter[0] > tcenter[2];
    bool gt12 = tcenter[1] > tcenter[2];

    // This is by far the fastest ways of indirectly sorting three values.
    // I measured it against std::sort() and three std::swap()s.
    axis[ gt01 +  gt02] = 0;
    axis[!gt01 +  gt12] = 1;
    axis[!gt02 + !gt12] = 2;

    // In axis, I now have indices into tcenter that defines sorting from smallest to biggest value.

    // Variable cell defines top left front coordinates of the current node, but I will reuse it to represent coordinates of the child cells.
    // The same is true for center - coordinates of the center of the current node is the same as top left front coordinate of the bottom right far child node.
    // Hope this diagrams clears the confusion:
    // cell---+------+
    // |      |      |
    // |      |      |
    // +----center---+
    // |      |      |
    // |      |      |
    // +------+------+
    // Now I want cell to be the coordinates of the first cell that can be theoretically intersected by the ray and center the last cell.
    // I do this by swapping coordinates according to ray direction sign.
    // This shall be precomputed in future version without recursion.
    for (uint8_t i = 0; i < 3; i++) {
      if (ray.direction[i] < 0.f) {
        std::swap(cell[i], center[i]);
      }
    }

    // I now know order of axes that are intersected by the ray, intervals of intersection for intersected child nodes and I know order of those child nodes according to ray direction.

    // I set the start of the new interval to the start of the current node interval
    float tmin = range.min;

    // The ray can intersect maximum of four child nodes.
    // The intervals are represented by:
    // <range.min, tcenter[axis[0]]>,
    // <tcenter[axis[0]], tcenter[axis[1]]>
    // <tcenter[axis[1]], tcenter[axis[2]]>
    // <tcenter[axis[2]], range.max>
    // Not all rays intersect four childs tho, so I need to check their validity.
    for (uint8_t i = 0; i < 3; i++) {
      // I compute if the intersection with the ith axis falls into the current node interval.
      float tmax = std::min(tcenter[axis[i]], range.max);

      if (tmin < tmax) {
        // If it does, i recurse knowing the full child interval and child cell.
        ray_octree_traversal(ray, { tmin, tmax }, cell, layer + 1, callback);

        // I advance the start of the interval.
        tmin = tmax;
      }

      // I traversed one axis, so i flip the coordinate on that axis to the other side.
      // This simulates xoring a bit in integer arithmetic, but in floats.
      cell[axis[i]] = center[axis[i]];
    }

    // This handles the last interval.
    // cell is == to center at this point because all three axes are flipped.
    if (tmin < range.max) {
      // Recurse the intersection checking
      ray_octree_traversal(ray, { tmin, range.max }, cell, layer + 1, callback);
    }
  }
}
