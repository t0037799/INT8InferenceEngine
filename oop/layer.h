#pragma once
#include <chrono>
#include <iostream>
#include <tuple>

#include "calibrator.h"
#include "mkl.h"
#include "pybind11/numpy.h"
#include "quant_utils.h"
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
        bias(Tensor<float>(out_channel)),
        q_weight(Tensor<s8_t>({out_channel, in_channel})),
        q_bias(Tensor<s8_t>(out_channel)) {}
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

  Tensor<u8_t>& forward_prop(Tensor<u8_t>&& in) {
    ssize_t m = in.shape()[0];
    ssize_t n = q_weight.shape()[0];
    ssize_t k = q_weight.shape()[1];
    Tensor<u8_t>* outp = new Tensor<u8_t>({m, n});
    Tensor<u8_t>& out = *outp;
    out.scale() = scale;
    out.zero_point() = zero_point;
    int* C = new int[m * n];
    int* oc = new int[n];
    for (int i = 0; i < n; ++i) {  // calculate offset after mul
      float t = 0;
      for (int j = 0; j < k; ++j) {
        t += in.zero_point() * q_weight(i, j);
      }
      oc[i] = -t;
    }
    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasRowOffset,
                       m, n, k, 1, in.data(), k, 0, q_weight.data(), k, 0, 0, C,
                       n, oc);
    for (ssize_t i = 0; i < m; ++i) {
      for (ssize_t j = 0; j < n; ++j) {
        C[i * n + j] += q_bias.data()[j] / in.scale();
      }
    }
    down_scale(out.data(), C, out.size(), in.scale(), q_weight.scale(),
               out.scale(), out.zero_point());
    delete[] C;
    delete[] oc;
    return out;
  }

  void load_weight(py::array_t<float> w) { weight.load_numpy(w); }
  void load_bias(py::array_t<float> b) { bias.load_numpy(b); }
  void prepare() {
    cal = new Calibrator();
    is_preparing = true;
  }
  void convert() {
    std::tie(scale, zero_point) = cal->get_range(1);
    delete cal;
    is_preparing = false;
    quantize();
    is_quantized = true;
  }

  void quantize() {
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

  Tensor<float> weight;
  Tensor<float> bias;
  Tensor<s8_t> q_weight;
  Tensor<s8_t> q_bias;
  Calibrator* cal;
  bool is_preparing = false;
  bool is_quantized = false;
  float scale;
  u8_t zero_point;
};

template <typename T>
Tensor<T>& relu(Tensor<T>&& in) {
  Tensor<T>* out = new Tensor<T>(in.shape());
  for (ssize_t i = 0; i < in.size(); ++i) {
    out->data()[i] = (in.data()[i] > 0) ? in.data()[i] : 0;
  }
  return *out;
}

template <>
Tensor<u8_t>& relu<u8_t>(Tensor<u8_t>&& in) {
  Tensor<u8_t>* out = new Tensor<u8_t>(in.shape());
  out->scale() = in.scale();
  out->zero_point() = in.zero_point();
  for (ssize_t i = 0; i < in.size(); ++i) {
    out->data()[i] =
        (in.data()[i] > in.zero_point()) ? in.data()[i] : in.zero_point();
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
  ssize_t oh = (h - kernel_size) / strides + 1;
  ssize_t ow = (w - kernel_size) / strides + 1;
  Tensor<T>* out = new Tensor<T>({n, c, oh, ow});
  out->scale() = in.scale();
  out->zero_point() = in.zero_point();
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      for (int k = 0, o = 0; o < oh; k += strides, ++o) {
        for (int l = 0, p = 0; p < ow; l += strides, ++p) {
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
