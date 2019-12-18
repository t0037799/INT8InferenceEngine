#pragma once
#include <chrono>
#include <iostream>
#include <tuple>

#include "calibrator.h"
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
    Tensor<float>* outp = new Tensor<float>({m, n});
    Tensor<float>& out = *outp;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, in.data(),
                k, weight.data(), k, 0, out.data(), n);
    for (ssize_t i = 0; i < m; ++i) {
      for (ssize_t j = 0; j < n; ++j) {
        out(i, j) += bias.data()[j];
      }
    }
    if (is_preparing) {
      cal->sample(out.data(), out.size());
    }
    return out;
  }

  void load_weight(py::array_t<float> w) { weight.load_numpy(w); }
  void load_bias(py::array_t<float> b) { bias.load_numpy(b); }
  void prepare() {
    cal = new Calibrator();
    is_preparing = true;
  }
  void convert() {
    range = cal->get_minmax(0.975);
    delete cal;
    is_preparing = false;
    quantize();
    is_quantized = true;
  }

  void quantize() {}

  Tensor<float> weight;
  Tensor<float> bias;
  Calibrator* cal;
  bool is_preparing = false;
  bool is_quantized = false;
  std::tuple<float, float> range;
};

template <typename T>
Tensor<T>& relu(Tensor<T>&& in) {
  Tensor<T>* out = new Tensor<T>(in.shape());
  for (ssize_t i = 0; i < in.size(); ++i) {
    out->data()[i] = (in.data()[i] > 0) ? in.data()[i] : 0;
  }
  return *out;
}

template <typename T>
T min() {
  return -std::numeric_limits<T>::max();
}
template <>
u8_t min<u8_t>() {
  return 0;
}
template <typename T>
Tensor<T>& maxpool2d(Tensor<T>&& in, ssize_t kernel_size, ssize_t strides) {
  auto shape = in.shape();
  auto [n, c, h, w] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
  Tensor<T>* out = new Tensor<T>(
      {n, c, (h - kernel_size) / strides + 1, (w - kernel_size) / strides + 1});
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      for (int k = 0, o = 0; k < h; k += strides, ++o) {
        for (int l = 0, p = 0; l < w; l += strides, ++p) {
          T r = min<T>();
          for (int m = 0; m < kernel_size; ++m) {
            for (int n = 0; n < kernel_size; ++n) {
              r = [](T a, T b) { return (a >= b) ? a : b; }(
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
