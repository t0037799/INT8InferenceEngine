#pragma once
#include "calibrator.h"
#include "mkl.h"
#include "pybind11/numpy.h"
#include "tensor.h"

class BaseLayer {
 public:
  BaseLayer(std::vector<ssize_t> weight_shape, ssize_t out_channel) {
    weight_ = std::unique_ptr<Tensor<float>>(new Tensor<float>(weight_shape));
    bias_ = std::make_unique<Tensor<float>>(out_channel);
  }
  template <typename T>
  Tensor<T>& forward_prop(Tensor<T>&& t) = delete;
  void load_weight(py::array_t<float> w) {
    if (weight_ == nullptr) {
      throw std::exception();
    }
    weight_->load_numpy(w);
  }
  void load_bias(py::array_t<float> b) {
    if (weight_ == nullptr) {
      throw std::exception();
    }
    bias_->load_numpy(b);
  }
  BaseLayer(Tensor<float>& w, Tensor<float>& b) {
    weight_ = std::make_unique<Tensor<float>>(w);
    bias_ = std::make_unique<Tensor<float>>(b);
  }
  BaseLayer(py::array_t<float> w, py::array_t<float> b) {
    weight_ = std::make_unique<Tensor<float>>(w);
    bias_ = std::make_unique<Tensor<float>>(b);
  }
  void prepare();
  void convert();

 protected:
  std::unique_ptr<Tensor<float>> weight_;
  std::unique_ptr<Tensor<float>> bias_;
  std::unique_ptr<Tensor<s8_t>> q_weight_;
  std::unique_ptr<Tensor<s8_t>> q_bias_;
  std::unique_ptr<Calibrator> cal_;
  bool is_preparing_ = false;
  bool is_quantized_ = false;
  float scale_ = 1;
  u8_t zero_point_ = 0;
};
