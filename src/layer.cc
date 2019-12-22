#include "layer.h"

#include <tuple>

#include "test_utils.h"
void quantize_weight(Tensor<s8_t>& q_weight, Tensor<s8_t>& q_bias,
                     Tensor<float>& weight, Tensor<float>& bias) {
  float max = -std::numeric_limits<float>::max();
  float min = std::numeric_limits<float>::max();
  for (ssize_t i = 0; i < weight.size(); ++i) {
    min = std::min(min, weight.data()[i]);
    max = std::max(max, weight.data()[i]);
  }
  for (ssize_t i = 0; i < bias.size(); ++i) {
    min = std::min(min, bias.data()[i]);
    max = std::max(max, bias.data()[i]);
  }
  q_weight.scale() = (max - min) / 127;
  q_bias.scale() = (max - min) / 127;
  for (ssize_t i = 0; i < weight.size(); ++i) {
    q_weight.data()[i] = weight.data()[i] / q_weight.scale();
  }
  for (ssize_t i = 0; i < bias.size(); ++i) {
    q_bias.data()[i] = bias.data()[i] / q_bias.scale();
  }
}

void BaseLayer::prepare() {
  if (is_quantized_) {
    print("already quantized");
    return;
  }
  cal_ = std::make_unique<Calibrator>();
  is_preparing_ = true;
}
void BaseLayer::convert() {
  if (is_quantized_) {
    print("already quantized");
    return;
  }
  if (is_preparing_ == false) {
    print("No prepared, use default config");
  } else {
    std::tie(scale_, zero_point_) = cal_->get_range(1);
    cal_.reset();
  }
  q_weight_ = std::make_unique<Tensor<s8_t>>(weight_->shape());
  q_bias_ = std::make_unique<Tensor<s8_t>>(bias_->shape());
  quantize_weight(*q_weight_, *q_bias_, *weight_, *bias_);
  is_preparing_ = false;
  is_quantized_ = true;
  weight_.reset();
  bias_.reset();
}
