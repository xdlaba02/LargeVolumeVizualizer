
#include <raw_volume/raw_volume.h>

#include <tree_volume/tree_volume.h>

#include <components/mapped_file.h>
#include <components/morton.h>
#include <components/endian.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

int main(int argc, char *argv[]) {
  if (argc != 7) {
    throw std::runtime_error("Invalid number of arguments!");
  }

  uint32_t width;
  uint32_t height;
  uint32_t depth;

  {
    std::stringstream wstream(argv[2]);
    std::stringstream hstream(argv[3]);
    std::stringstream dstream(argv[4]);
    wstream >> width;
    hstream >> height;
    dstream >> depth;
  }

  RawVolume<uint8_t> volume(argv[1], width, height, depth);

  TreeVolume<uint8_t>::Info info(width, height, depth);

  {
    std::ofstream metadata(argv[6], std::ios::binary);
    if (!metadata) {
      throw std::runtime_error(std::string("Unable to open '") + argv[6] + "'!");
    }

    metadata.seekp(info.size_in_blocks * sizeof(TreeVolume<uint8_t>::Node) - 1);
    metadata.write("", 1);
  }

  MappedFile metadata(argv[6], 0, info.size_in_blocks * sizeof(TreeVolume<uint8_t>::Node), MappedFile::WRITE, MappedFile::SHARED);
  if (!metadata) {
    throw std::runtime_error(std::string("Unable to map '") + argv[6] + "'!");
  }

  TreeVolume<uint8_t>::Node *nodes = reinterpret_cast<TreeVolume<uint8_t>::Node *>(metadata.data());

  std::ofstream tree_volume(argv[5], std::ofstream::binary);
  if (!tree_volume) {
    throw std::runtime_error(std::string("Unable to open '") + argv[5] + "'!");
  }

  uint64_t block_handle = 0;

  #pragma omp parallel for
  for (uint32_t block_z = 0; block_z < info.layers[0].depth_in_blocks; block_z++) {
    for (uint32_t block_y = 0; block_y < info.layers[0].height_in_blocks; block_y++) {
      for (uint32_t block_x = 0; block_x < info.layers[0].width_in_blocks; block_x++) {

        TreeVolume<uint8_t>::Node &node = nodes[info.node_handle(block_x, block_y, block_z, 0)];

        node.block_handle = 0;
        node.min    = 255;
        node.max    = 0;

        TreeVolume<uint8_t>::Block block {};

        for (uint32_t i = 0; i < TreeVolume<uint8_t>::BLOCK_SIZE; i++) {
          uint32_t x, y, z;
          Morton<TreeVolume<uint8_t>::BLOCK_BITS>::from_index(i, x, y, z);

          uint32_t original_x = std::min(block_x * TreeVolume<uint8_t>::SUBVOLUME_SIDE + x, width - 1);
          uint32_t original_y = std::min(block_y * TreeVolume<uint8_t>::SUBVOLUME_SIDE + y, height - 1);
          uint32_t original_z = std::min(block_z * TreeVolume<uint8_t>::SUBVOLUME_SIDE + z, depth - 1);

          block[i] = volume.data[volume.voxel_handle(original_x, original_y, original_z)];

          node.min = std::min<uint8_t>(node.min, block[i]);
          node.max = std::max<uint8_t>(node.max, block[i]);
        }

        if (node.min != node.max) {
          #pragma omp critical
          {
            node.block_handle = block_handle++;
            tree_volume.write((const char *)block, sizeof(block));
          }
        }
      }
    }
  }

  for (uint8_t layer = 1; layer < std::size(info.layers); layer++) {
    tree_volume.flush();

    MappedFile blocks_file(argv[5], 0, tree_volume.tellp(), MappedFile::READ, MappedFile::SHARED);

    if (!blocks_file) {
      throw std::runtime_error(std::string("Unable to map '") + argv[5] + "'!");
    }

    const TreeVolume<uint8_t>::Block *blocks = reinterpret_cast<const TreeVolume<uint8_t>::Block *>(blocks_file.data());

    #pragma omp parallel for
    for (uint32_t block_z = 0; block_z < info.layers[layer].depth_in_blocks; block_z++) {

      uint32_t block_start_z = block_z * TreeVolume<uint8_t>::SUBVOLUME_SIDE;

      for (uint32_t block_y = 0; block_y < info.layers[layer].height_in_blocks; block_y++) {

        uint32_t block_start_y = block_y * TreeVolume<uint8_t>::SUBVOLUME_SIDE;

        for (uint32_t block_x = 0; block_x < info.layers[layer].width_in_blocks; block_x++) {

          uint32_t block_start_x = block_x * TreeVolume<uint8_t>::SUBVOLUME_SIDE;

          TreeVolume<uint8_t>::Node &node = nodes[info.node_handle(block_x, block_y, block_z, layer)];

          node.block_handle = 0;
          node.min    = 255;
          node.max    = 0;

          TreeVolume<uint8_t>::Block block {};

          for (uint32_t in_block_z = 0; in_block_z < TreeVolume<uint8_t>::BLOCK_SIDE; in_block_z++) {

            uint32_t voxel_z = block_start_z + in_block_z;
            uint32_t original_voxel_z = voxel_z << 1;

            for (uint32_t in_block_y = 0; in_block_y < TreeVolume<uint8_t>::BLOCK_SIDE; in_block_y++) {

              uint32_t voxel_y = block_start_y + in_block_y;
              uint32_t original_voxel_y = voxel_y << 1;

              for (uint32_t in_block_x = 0; in_block_x < TreeVolume<uint8_t>::BLOCK_SIDE; in_block_x++) {

                uint32_t voxel_x = block_start_x + in_block_x;
                uint32_t original_voxel_x = voxel_x << 1;


                uint32_t original_block_z = std::min(original_voxel_z / TreeVolume<uint8_t>::SUBVOLUME_SIDE, info.layers[layer - 1].depth_in_blocks  - 1);
                uint32_t original_in_block_z = std::min(original_voxel_z - original_block_z * TreeVolume<uint8_t>::SUBVOLUME_SIDE, TreeVolume<uint8_t>::SUBVOLUME_SIDE - 1);

                uint32_t original_block_y = std::min(original_voxel_y / TreeVolume<uint8_t>::SUBVOLUME_SIDE, info.layers[layer - 1].height_in_blocks - 1);
                uint32_t original_in_block_y = std::min(original_voxel_y - original_block_y * TreeVolume<uint8_t>::SUBVOLUME_SIDE, TreeVolume<uint8_t>::SUBVOLUME_SIDE - 1);

                uint32_t original_block_x = std::min(original_voxel_x / TreeVolume<uint8_t>::SUBVOLUME_SIDE, info.layers[layer - 1].width_in_blocks  - 1);
                uint32_t original_in_block_x = std::min(original_voxel_x - original_block_x * TreeVolume<uint8_t>::SUBVOLUME_SIDE, TreeVolume<uint8_t>::SUBVOLUME_SIDE - 1);

                TreeVolume<uint8_t>::Node &original_node = nodes[info.node_handle(original_block_x, original_block_y, original_block_z, layer - 1)];

                uint32_t value;
                if (original_node.min != original_node.max) {
                  value = blocks[original_node.block_handle][Morton<TreeVolume<uint8_t>::BLOCK_BITS>::to_index(original_in_block_x, original_in_block_y, original_in_block_z)];
                }
                else {
                  value = original_node.min;
                }

                node.min = std::min<uint32_t>(node.min, value);
                node.max = std::max<uint32_t>(node.max, value);

                block[Morton<TreeVolume<uint8_t>::BLOCK_BITS>::to_index(in_block_x, in_block_y, in_block_z)] = value;
              }
            }
          }

          for (uint32_t original_block_z_unaligned = (block_z << 1); original_block_z_unaligned < (block_z << 1) + 2; original_block_z_unaligned++) {

            uint32_t original_block_z = std::min(original_block_z_unaligned, info.layers[layer - 1].depth_in_blocks  - 1);

            for (uint32_t original_block_y_unaligned = (block_y << 1); original_block_y_unaligned < (block_y << 1) + 2; original_block_y_unaligned++) {

              uint32_t original_block_y = std::min(original_block_y_unaligned, info.layers[layer - 1].height_in_blocks - 1);

              for (uint32_t original_block_x_unaligned = (block_x << 1); original_block_x_unaligned < (block_x << 1) + 2; original_block_x_unaligned++) {

                uint32_t original_block_x = std::min(original_block_x_unaligned, info.layers[layer - 1].width_in_blocks  - 1);

                TreeVolume<uint8_t>::Node &original_node = nodes[info.node_handle(original_block_x, original_block_y, original_block_z, layer - 1)];

                node.min = std::min(node.min, original_node.min);
                node.max = std::max(node.max, original_node.max);
              }
            }
          }

          if (node.min != node.max) {
            #pragma omp critical
            {
              node.block_handle = block_handle++;
              tree_volume.write((const char *)block, sizeof(block));
            }
          }
        }
      }
    }
  }
}
