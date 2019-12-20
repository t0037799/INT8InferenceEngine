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
void im2col_tile(T* M, const T* I, int c, int h, int w, int kh, int kw, int i,
                 int j, int zero_point) {
  for (int k = 0; k < c; ++k) {
    for (int l = 0; l < kh; ++l) {
      for (int m = 0; m < kw; ++m) {
        const ssize_t idx = (h * w) * k + (i + l) * w + (j + m);
        if ((i + l) < 0 || (j + m) < 0 || (i + l) >= h || (j + m) >= w) {
          M[k * kh * kw + l * kw + m] = zero_point;
        } else {
          M[k * kh * kw + l * kw + m] = I[idx];
        }
      }
    }
  }
}

template <typename T>
void im2col(T* M, const T* I, int c, int h, int w, int kh, int kw,
            ssize_t stride, ssize_t padding, int zero_point) {
  ssize_t oh = (h - kh + 2 * padding) / stride + 1;
  ssize_t ow = (w - kw + 2 * padding) / stride + 1;
  for (int i = -padding, ti = 0; ti < oh; i += stride, ++ti) {
    for (int j = -padding, tj = 0; tj < ow; j += stride, ++tj) {
      if (padding != 0) {
        im2col_tile(M + (ti * ow + tj) * c * kh * kw, I, c, h, w, kh, kw, i, j,
                    zero_point);
      } else {
        im2col_tile(M + (ti * ow + tj) * c * kh * kw, I, c, h, w, kh, kw, i, j);
      }
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
  Conv2d(ssize_t in_channel, ssize_t out_channel, ssize_t kernel_size,
         ssize_t stride = 1, ssize_t padding = 0)
      : weight(
            Tensor<float>({out_channel, in_channel, kernel_size, kernel_size})),
        bias(Tensor<float>(out_channel)),
        q_weight(
            Tensor<s8_t>({out_channel, in_channel, kernel_size, kernel_size})),
        q_bias(Tensor<s8_t>(out_channel)),
        stride(stride),
        padding(padding) {
    if (stride == 0) {
      throw std::exception();
    }
  }
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
    ssize_t oh = (h - kh + 2 * padding) / stride + 1;
    ssize_t ow = (w - kw + 2 * padding) / stride + 1;
    Tensor<float>* outp = new Tensor<float>({n, kc, oh, ow});
    Tensor<float>& out = *outp;
    ssize_t mat_m = kc;
    ssize_t mat_n = oh * ow;
    ssize_t mat_k = c * kh * kw;
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      float* matricize = new float[mat_k * mat_n];
      im2col(matricize, &in(i, 0, 0, 0), c, h, w, kh, kw, stride, padding, 0);
      float* C = const_cast<float*>(&out(i, 0, 0, 0));
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mat_m, mat_n, mat_k,
                  1, weight.data(), mat_k, matricize, mat_k, 0, C, mat_n);
      for (int j = 0; j < kc; ++j) {
        for (int k = 0; k < oh; ++k) {
          for (int l = 0; l < ow; ++l) {
            out(i, j, k, l) += bias.data()[j];
          }
        }
      }
      delete[] matricize;
    }
    if (is_preparing) {
      cal->sample(out.data(), out.size());
    }
    return out;
  }

  Tensor<float>& forward_prop2(Tensor<float>&& in) {
    ssize_t n = in.shape()[0];
    ssize_t c = in.shape()[1];
    ssize_t h = in.shape()[2];
    ssize_t w = in.shape()[3];
    ssize_t kc = weight.shape()[0];
    ssize_t kh = weight.shape()[2];
    ssize_t kw = weight.shape()[3];
    ssize_t oh = (h - kh + 2 * padding) / stride + 1;
    ssize_t ow = (w - kw + 2 * padding) / stride + 1;
    Tensor<float>* outp = new Tensor<float>({n, kc, oh, ow});
    Tensor<float>& out = *outp;
    ssize_t mat_m = oh * ow;
    ssize_t mat_n = kc;
    ssize_t mat_k = c * kh * kw;
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      float* matricize = new float[mat_m * mat_k];
      im2col(matricize, &in(i, 0, 0, 0), c, h, w, kh, kw, stride, padding, 0);
      float* C = const_cast<float*>(&out(i, 0, 0, 0));
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mat_m, mat_n, mat_k,
                  1, matricize, mat_k, weight.data(), mat_k, 0, C, mat_n);
      for (int j = 0; j < kc; ++j) {
        for (int k = 0; k < mat_m; ++k) {
          C[k * kc + j] += bias.data()[j];
        }
      }
      transpose(&out(i, 0, 0, 0), mat_m, mat_n);
      delete[] matricize;
    }
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
    ssize_t kc = q_weight.shape()[0];
    ssize_t kh = q_weight.shape()[2];
    ssize_t kw = q_weight.shape()[3];
    ssize_t oh = (h - kh + 2 * padding) / stride + 1;
    ssize_t ow = (w - kw + 2 * padding) / stride + 1;
    Tensor<u8_t>* outp = new Tensor<u8_t>({n, kc, oh, ow});
    Tensor<u8_t>& out = *outp;
    out.scale() = scale;
    out.zero_point() = zero_point;
    ssize_t mat_m = oh * ow;
    ssize_t mat_n = kc;
    ssize_t mat_k = c * kh * kw;
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      u8_t* matricize = new u8_t[mat_m * mat_k];
      int* C = new int[mat_m * mat_n];
      int* oc = new int[mat_n];
      for (int j = 0; j < mat_n; ++j) {  // calculate offset after mul
        float t = 0;
        for (int k = 0; k < mat_k; ++k) {
          t += in.zero_point() * q_weight.data()[j * mat_k + k];
        }
        oc[j] = q_bias.data()[j] / in.scale() - t;  // bias count in offset
      }
      im2col(matricize, &in(i, 0, 0, 0), c, h, w, kh, kw, stride, padding,
             in.zero_point());
      cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans,
                         CblasRowOffset, mat_m, mat_n, mat_k, 1, matricize,
                         mat_k, 0, q_weight.data(), mat_k, 0, 0, C, mat_n, oc);
      down_scale(&out(i, 0, 0, 0), C, mat_m * mat_n, in.scale(),
                 q_weight.scale(), out.scale(), out.zero_point());
      transpose(&out(i, 0, 0, 0), mat_m, mat_n);
      delete[] oc;
      delete[] C;
      delete[] matricize;
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
  ssize_t padding;
  ssize_t stride;
  Calibrator* cal;
  bool is_preparing = false;
  bool is_quantized = false;
  float scale;
  u8_t zero_point;
};
