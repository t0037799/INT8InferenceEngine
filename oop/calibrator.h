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
  std::tuple<float, float> get_minmax(float quantile) {
    std::sort(out_samples.begin(), out_samples.end());
    float out_min = out_samples[(1.0 - quantile) * out_cnt];
    float out_max = out_samples[quantile * (out_cnt - 1)];
    return std::make_tuple(out_min, out_max);
  }
  std::array<float, num_samples> out_samples;
  ssize_t out_cnt;
};
