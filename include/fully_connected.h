#pragma once
#include "layer.h"

class Linear : public BaseLayer {
 public:
  Linear() = delete;
  Linear(ssize_t in_channel, ssize_t out_channel)
      : BaseLayer({out_channel, in_channel}, out_channel) {}
  Linear(Tensor<float>& w, Tensor<float>& b) : BaseLayer(w, b) {}
  Linear(py::array_t<float> w, py::array_t<float> b) : BaseLayer(w, b) {}
  Tensor<float>& forward_prop(Tensor<float>&& in);
  Tensor<u8_t>& forward_prop(Tensor<u8_t>&& in);
};
