
#include "../tools/tf1d.h"
#include <transfer/preintegrate_function.h>
#include <transfer/piecewise_linear.h>

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    return 1;
  }

  TF1D tf1d = TF1D::load_from_file(argv[1]);

  static constinit uint32_t pre_size = 256;

  Texture2D<float> transfer_r = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.rgb, v * 256 / pre_size).r; });
  Texture2D<float> transfer_g = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.rgb, v * 256 / pre_size).g; });
  Texture2D<float> transfer_b = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.rgb, v * 256 / pre_size).b; });
  Texture2D<float> transfer_a = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.a, v * 256 / pre_size); });

  std::ofstream out(argv[2]);
  if (out) {
    for (size_t y = 0; y < pre_size; y++) {
      for (size_t x = 0; x < pre_size; x++) {
        out.put(transfer_r(x, y) * 255);
        out.put(transfer_g(x, y) * 255);
        out.put(transfer_b(x, y) * 255);
        out.put(transfer_a(x, y) * 255);
      }
    }
  }

  return 0;
}
