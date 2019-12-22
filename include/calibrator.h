#pragma once
#include "tensor.h"

constexpr ssize_t num_samples = 1000;
class Calibrator {
 public:
  Calibrator() = default;
  void sample(float* out_data, ssize_t out_size);
  std::tuple<float, u8_t> get_range(float quantile);

 private:
  std::array<float, num_samples> out_samples;
  ssize_t out_cnt = 0;
};
