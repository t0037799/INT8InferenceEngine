#include <random>

#include "tensor.h"

constexpr ssize_t num_samples = 1000;
class Calibrator {
 public:
  void sample(
      float* out_data,
      ssize_t out_size) {  // if get sampled random replace if array is full
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<ssize_t> dist(
        0, num_samples * 2);  // half chance get sampled
    for (ssize_t i = 0; i < out_size; ++i) {
      if (out_cnt < num_samples) {
        out_samples[out_cnt++] = out_data[i];
      } else {
        ssize_t idx = dist(rng);
        if (idx < num_samples) {
          out_samples[idx] = out_data[i];
        }
      }
    }
  }
  std::tuple<float, u8_t> get_range(float quantile) {
    std::sort(out_samples.begin(), out_samples.end());
    float out_min = out_samples[(1.0 - quantile) * out_cnt];
    float out_max = out_samples[quantile * (out_cnt - 1)];
    out_min = std::fmin(out_min, 0.);
    out_max = std::fmax(out_max, 0.);
    u8_t zero_point = 255 * (0 - out_min) / (out_max - out_min + 1e-09);
    float scale = (zero_point == 0) ? (out_max - out_min) / 255
                                    : (0 - out_min) / zero_point;
    if (scale == 0) {  // not sample or edge case
      scale = 1;       // default;
    }
    return std::make_tuple(scale, zero_point);
  }
  std::array<float, num_samples> out_samples;
  ssize_t out_cnt;
};
