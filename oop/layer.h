#pragma once
#include <chrono>
#include <iostream>

#include "mkl.h"
#include "pybind11/numpy.h"
#include "tensor.h"

class ILayer {
 public:
  virtual Tensor<float>& forward_prop(Tensor<float>&& t) = 0;
  virtual ~ILayer() {}
};

class Linear : ILayer {
 public:
  Linear() = delete;
  Linear(ssize_t in_channel, ssize_t out_channel)
      : weight(Tensor<float>({in_channel, out_channel})),
        bias(Tensor<float>(out_channel)) {}
  Linear(Tensor<float> _w, Tensor<float> _b) : weight(_w), bias(_b) {}
  Linear(py::array_t<float> _w, py::array_t<float> _b)
      : weight(Tensor<float>(_w)), bias(Tensor<float>(_b)) {}
  Tensor<float>& forward_prop(Tensor<float>&& in) {
    ssize_t m = in.shape()[0];
    ssize_t n = weight.shape()[0];
    ssize_t k = weight.shape()[1];
    Tensor<float>* out = new Tensor<float>({m, n});
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, in.data(),
                k, weight.data(), k, 0, out->data(), n);
    for (ssize_t i = 0; i < m; ++i) {
      for (ssize_t j = 0; j < n; ++j) {
        out->data()[i * n + j] += bias.data()[j];
      }
    }
    return *out;
  }

  void load_weight(py::array_t<float> _w, py::array_t<float> _b) {
    weight.load_numpy(_w);
    bias.load_numpy(_b);
  }

  Tensor<float> weight;
  Tensor<float> bias;
};
