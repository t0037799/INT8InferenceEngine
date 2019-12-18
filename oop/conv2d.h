#include "layer.h"

void im2col_tile(float* M, const float* I, int c, int h, int w, int kh, int kw,
                 int i, int j) {
  for (int k = 0; k < c; ++k) {
    for (int l = 0; l < kh; ++l) {
      for (int m = 0; m < kw; ++m) {
        M[k * kh * kw + l * kw + m] = I[(h * w) * k + (i + l) * w + (j + m)];
      }
    }
  }
}

void im2col(float* M, const float* I, int c, int h, int w, int kh, int kw) {
  for (int i = 0; i < h - kh + 1; ++i) {
    for (int j = 0; j < w - kw + 1; ++j) {
      im2col_tile(M + (i * (w - kw + 1) + j) * c * kh * kw, I, c, h, w, kh, kw,
                  i, j);
    }
  }
}

class Conv2d : ILayer {
 public:
  Conv2d() = delete;
  Conv2d(ssize_t in_channel, ssize_t out_channel, ssize_t kernel_size)
      : weight(
            Tensor<float>({out_channel, in_channel, kernel_size, kernel_size})),
        bias(Tensor<float>(out_channel)) {}
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
    is_quantized = true;
  }

  Tensor<float> weight;
  Tensor<float> bias;
  Calibrator* cal;
  bool is_preparing = false;
  bool is_quantized = false;
  std::tuple<float, float> range;
};
