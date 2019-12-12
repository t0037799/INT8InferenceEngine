#pragma once
#include <chrono>
#include <iostream>
#include <tuple>

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
      : weight(Tensor<float>({out_channel, in_channel})),
        bias(Tensor<float>(out_channel)) {}
  Linear(Tensor<float> _w, Tensor<float> _b) : weight(_w), bias(_b) {}
  Linear(py::array_t<float> _w, py::array_t<float> _b)
      : weight(Tensor<float>(_w)), bias(Tensor<float>(_b)) {}
  ~Linear() {
    // std::cerr << "bye linear" << bias.size() << "\n";
  }
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

  void load_weight(py::array_t<float> w) { weight.load_numpy(w); }
  void load_bias(py::array_t<float> b) { bias.load_numpy(b); }

  Tensor<float> weight;
  Tensor<float> bias;
};

class Relu : ILayer {
 public:
  Tensor<float>& forward_prop(Tensor<float>&& in) {
    Tensor<float>* out = new Tensor<float>(in.shape());
    for (ssize_t i = 0; i < in.size(); ++i) {
      out->data()[i] = (in.data()[i] > 0) ? in.data()[i] : 0;
    }
    return *out;
  }
};

class Maxpool2d : ILayer {
 public:
  Maxpool2d() = delete;
  Maxpool2d(ssize_t kernel_size, ssize_t strides)
      : kernel_size(kernel_size), strides(strides) {}
  Tensor<float>& forward_prop(Tensor<float>&& in) {
    auto shape = in.shape();
    auto [n, c, h, w] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
    Tensor<float>* out =
        new Tensor<float>({n, c, (h - kernel_size) / strides + 1,
                           (w - kernel_size) / strides + 1});
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < c; ++j) {
        for (int k = 0, o = 0; k < h; k += strides, ++o) {
          for (int l = 0, p = 0; l < w; l += strides, ++p) {
            float r = -std::numeric_limits<float>::max();
            for (int m = 0; m < kernel_size; ++m) {
              for (int n = 0; n < kernel_size; ++n) {
                r = [](float a, float b) { return (a >= b) ? a : b; }(
                        r, in(i, j, k + m, l + n));
                // r = std::max(r, in(i, j, k + m, l + n)); it's slower
              }
            }
            (*out)(i, j, o, p) = r;
          }
        }
      }
    }
    return *out;
  }

  ssize_t kernel_size;
  ssize_t strides;
};
