#pragma once

#include <getopt.h>

#include <cstdint>
#include <cstddef>

#include <sstream>
#include <iostream>

void print_usage(const char *argv0) {
  std::cerr << "Usage: \n";
  std::cerr << argv0 << " <processed-volume> <processed-metadata> <transfer-function> <width> <height> <depth>\n";
}

bool parse_args(int argc, char *argv[], const char *&processed_volume, const char *&processed_metadata, const char *&transfer_function, uint32_t &width, uint32_t &height, uint32_t &depth) {
  if (argc != 7) {
    print_usage(argv[0]);
    return false;
  }

  processed_volume   = argv[1];
  processed_metadata = argv[2];
  transfer_function  = argv[3];

  {
    std::stringstream arg {};
    arg << argv[4];
    arg >> width;
    if (!arg) {
      print_usage(argv[0]);
      return false;
    }
  }

  {
    std::stringstream arg {};
    arg << argv[5];
    arg >> height;
    if (!arg) {
      print_usage(argv[0]);
      return false;
    }
  }

  {
    std::stringstream arg {};
    arg << argv[6];
    arg >> depth;
    if (!arg) {
      print_usage(argv[0]);
      return false;
    }
  }

  return true;
}
