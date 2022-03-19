#pragma once

#include "tree_volume.h"
#include "sampler.h"

#include <ray_traversal/raster.h>
#include <ray_traversal/intersection.h>

#include <utils/utils.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
glm::vec4 integrate(const TreeVolume<T> &volume, const Ray &ray, uint8_t layer, float step, const TransferFunctionType &transfer_function) {
  float tmin {};
  float tmax {};

  intersect_aabb_ray(ray.origin, ray.direction_inverse, {0.f, 0.f, 0.f}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, tmin, tmax);

  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  if (tmin < tmax) {

    // First slab in infinitely small because we want to initialize start value with first encountered value without producing output
    float slab_start_t = tmin;
    float slab_end_t = tmin;
    float slab_start_value = 0.f;

    float stepsize = step * approx_exp2(layer);

    ray_raster_traversal(ray, { tmin, tmax }, { 0.f, 0.f, 0.f }, 0, [&](const RayRange &range, const glm::vec3 &cell) {

      // Early ray termination
      if (dst.a < 0.01f) {
        return false;
      }

      // Next step does not intersect the block
      if (slab_end_t >= range.max) {
        return true;
      }

      if (cell.x >= volume.info.layers[layer].width_in_blocks
      || (cell.y >= volume.info.layers[layer].height_in_blocks)
      || (cell.z >= volume.info.layers[layer].depth_in_blocks)) {
        slab_start_t = range.max;
        slab_end_t = range.max;
        return true;
      }

      const auto &node = volume.nodes[volume.info.node_handle(cell.x, cell.y, cell.z, layer)];

      auto node_rgba = transfer_function(node.min, node.max);

      // Empty space skipping
      if (node_rgba.a == 0.f) {
        blend(transfer_function(slab_start_value, node.min), dst, slab_end_t - slab_start_t); // finish previous step with block value

        slab_start_t = range.max;
        slab_end_t = range.max;

        return true;
      }

      // Fast integration of uniform space
      if (node.min == node.max) {
        blend(transfer_function(slab_start_value, node.min), dst, slab_end_t - slab_start_t); // finish previous step with block value
        blend(node_rgba, dst, range.max - slab_end_t); // blend the rest of the block

        slab_start_value = node.min;
        slab_start_t = range.max;
        slab_end_t = range.max + stepsize;

        return true;
      }

      // Numeric integration
      while (slab_end_t < range.max) {
        glm::vec3 pos = ray.origin + ray.direction * slab_end_t;

        glm::vec3 in_block = (pos - cell) * approx_exp2(layer) * float(TreeVolume<T>::SUBVOLUME_SIDE);

        Samplet sampl = samplet(volume, node.block_handle, in_block.x, in_block.y, in_block.z);

        float slab_end_value = linterp(sampl);

        glm::vec4 src = transfer_function(slab_start_value, slab_end_value);

        blend(src, dst, slab_end_t - slab_start_t);

        slab_start_value = slab_end_value;
        slab_start_t = slab_end_t;
        slab_end_t = slab_end_t + stepsize;
      }

      return true;
    });
  }

  return dst;
}
