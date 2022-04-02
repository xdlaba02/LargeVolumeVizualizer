
#include "sample.h"
#include "integrate.h"
#include "timer.h"

#include <ray/ray.h>
#include <ray/ray_simd.h>
#include <ray/intersection.h>
#include <ray/intersection_simd.h>

#include <utils/mapped_file.h>
#include <utils/ray_generator.h>

#include <cstdint>

#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    throw std::runtime_error("Param cnt");
  }

  uint64_t side;

  {
    std::stringstream arg {};
    arg << argv[1];
    arg >> side;
    if (!arg) {
      throw std::runtime_error("Param type");
    }
  }

  {
    std::ofstream tmp(argv[2], std::ios::binary);
    if (!tmp) {
      throw std::runtime_error(std::string("Unable to open '") + argv[2] + "'!");
    }

    tmp.seekp(side * side * side - 1);
    tmp.write("", 1);
  }

  MappedFile tmp(argv[2], 0, side * side * side, MappedFile::READ, MappedFile::SHARED);
  if (!tmp) {
    throw std::runtime_error(std::string("Unable to map '") + argv[2] + "'!");
  }

  float dummy {};

  {
    glm::vec3 ray_origin = glm::vec4(0.f, 0.f, 1.f, 1.f);

    RayGenerator ray_generator(1024, 1024, 45.f);

    size_t nsamples = 0;

    std::cerr << measure_ns([&]{
      for (uint32_t y = 0; y < 1024; y++) {
        for (uint32_t x = 0; x < 1024; x++) {

          glm::vec3 dir = ray_generator(x, y);

          integrate_scalar({ ray_origin, dir, 1.f / dir }, 0.001, [&](const glm::vec3 &pos) {
            dummy += sample_raw_scalar(reinterpret_cast<uint8_t *>(tmp.data()), side, side, side, pos.x, pos.y, pos.z);
            nsamples++;
          });
        }
      }
    }) / nsamples << " ns per sample\n";
  }

  std::cerr << dummy;

  return 0;
}
