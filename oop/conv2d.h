#include "layer.h"
#include "tensor.h"

template <typename T>
void im2col_tile(T* M, const T* I, int c, int h, int w, int kh, int kw, int i,
                 int j) {
  for (int k = 0; k < c; ++k) {
    for (int l = 0; l < kh; ++l) {
      for (int m = 0; m < kw; ++m) {
        M[k * kh * kw + l * kw + m] = I[(h * w) * k + (i + l) * w + (j + m)];
      }
    }
  }
}

template <typename T>
void im2col(T* M, const T* I, int c, int h, int w, int kh, int kw) {
  for (int i = 0; i < h - kh + 1; ++i) {
    for (int j = 0; j < w - kw + 1; ++j) {
      im2col_tile(M + (i * (w - kw + 1) + j) * c * kh * kw, I, c, h, w, kh, kw,
                  i, j);
    }
  }
}

template <typename T>
void transpose(T* data, ssize_t nrow, ssize_t ncol) {
  T* back_data = new T[nrow * ncol];
  std::memcpy(back_data, data, nrow * ncol * sizeof(T));
  for (ssize_t i = 0; i < nrow; ++i) {
    for (ssize_t j = 0; j < ncol; ++j) {
      data[j * nrow + i] = back_data[i * ncol + j];
    }
  }
  delete[] back_data;
}

class Conv2d : ILayer {
 public:
  Conv2d() = delete;
  Conv2d(ssize_t in_channel, ssize_t out_channel, ssize_t kernel_size)
      : weight(
            Tensor<float>({out_channel, in_channel, kernel_size, kernel_size})),
        bias(Tensor<float>(out_channel)),
        q_weight(
            Tensor<s8_t>({out_channel, in_channel, kernel_size, kernel_size})),
        q_bias(Tensor<s8_t>(out_channel)) {}
  Conv2d(Tensor<float> _w, Tensor<float> _b) : weight(_w), bias(_b) {}
  Conv2d(py::array_t<float> _w, py::array_t<float> _b)
      : weight(Tensor<float>(_w)), bias(Tensor<float>(_b)) {}
  ~Conv2d() {
    // std::cerr << "bye conv2d" << bias.size() << "\n";
  }
  Tensor<float>& forward_prop(Tensor<float>&& in) {
    ssize_t n = in.shape()[0];
    ssize_t c = in.shape()[1];
    ssize_t h = in.shape()[2];
    ssize_t w = in.shape()[3];
    ssize_t kc = weight.shape()[0];
    ssize_t kh = weight.shape()[2];
    ssize_t kw = weight.shape()[3];
    Tensor<float>* outp = new Tensor<float>({n, kc, h - kh + 1, w - kw + 1});
    Tensor<float>& out = *outp;
    ssize_t matrix_sz = (h - kh + 1) * (w - kw + 1) * c * kh * kw;
    float* matricize = new float[n * matrix_sz];
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      im2col(matricize + i * matrix_sz, &in(i, 0, 0, 0), c, h, w, kh, kw);
      float* C = const_cast<float*>(&out(i, 0, 0, 0));
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kc,
                  (h - kh + 1) * (w - kw + 1), c * kh * kw, 1, weight.data(),
                  c * kh * kw, matricize + i * matrix_sz, c * kh * kw, 0, C,
                  (h - kh + 1) * (w - kw + 1));
      for (int j = 0; j < kc; ++j) {
        for (int k = 0; k < (h - kh + 1); ++k) {
          for (int l = 0; l < (w - kw + 1); ++l) {
            out(i, j, k, l) += bias.data()[j];
          }
        }
      }
    }
    delete[] matricize;
    if (is_preparing) {
      cal->sample(out.data(), out.size());
    }

    return out;
  }

  Tensor<u8_t>& forward_prop(Tensor<u8_t>&& in) {
    ssize_t n = in.shape()[0];
    ssize_t c = in.shape()[1];
    ssize_t h = in.shape()[2];
    ssize_t w = in.shape()[3];
    ssize_t kc = weight.shape()[0];
    ssize_t kh = weight.shape()[2];
    ssize_t kw = weight.shape()[3];
    Tensor<u8_t>* outp = new Tensor<u8_t>({n, kc, h - kh + 1, w - kw + 1});
    Tensor<u8_t>& out = *outp;
    ssize_t matrix_sz = (h - kh + 1) * (w - kw + 1) * c * kh * kw;
    u8_t* matricize = new u8_t[n * matrix_sz];
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      im2col(matricize + i * matrix_sz, &in(i, 0, 0, 0), c, h, w, kh, kw);
      int* C = new int[kc * (h - kh + 1) * (w - kw + 1)];
      /*
  cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans,
                     (h - kh + 1) * (w - kw + 1), kc, c * kh * kw, 1,
                     matricize + i * matrix_sz, c * kh * kw, weight.data(),
                     c * kh * kw, 0, C, kc);
                                             */
      for (int l = 0; l < (w - kw + 1); ++l) {
        for (int k = 0; k < (h - kh + 1); ++k) {
          for (int j = 0; j < kc; ++j) {
            C[l * (h - kh + 1) * kc + k * kc + j] += bias.data()[j];
          }
        }
      }
      // dequantize(C, scale, zero_point);
      transpose(C, (h - kh + 1) * (w - kw + 1), kc);
      delete[] C;
    }
    delete[] matricize;
    return out;
  }
  void load_weight(py::array_t<float> w) { weight.load_numpy(w); }
  void load_bias(py::array_t<float> b) { bias.load_numpy(b); }
  void prepare() {
    cal = new Calibrator();
    is_preparing = true;
  }
  void convert() {
    auto [scale, zero_point] = cal->get_range(0.975);
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
