#include <limits>

#include "mkl.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

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

py::array_t<float> conv2d(py::array_t<float> weight, py::array_t<float> bias,
                          py::array_t<float> in) {
  auto wi = weight.unchecked<4>();
  auto ii = in.unchecked<4>();
  auto bi = bias.unchecked<1>();
  int n = ii.shape(0);
  int c = ii.shape(1);
  int h = ii.shape(2);
  int w = ii.shape(3);
  int kc = wi.shape(0);
  int kh = wi.shape(2);
  int kw = wi.shape(3);
  py::array_t<float> res({n, kc, h - kh + 1, w - kw + 1});
  auto ri = res.mutable_unchecked<4>();
  int matrix_sz = (h - kh + 1) * (w - kw + 1) * c * kh * kw;
  float* matricize = new float[n * matrix_sz];
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    im2col(matricize + i * matrix_sz, ii.data(i, 0, 0, 0), c, h, w, kh, kw);
    float* C = const_cast<float*>(res.data(i, 0, 0, 0));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, kc,
                (h - kh + 1) * (w - kw + 1), c * kh * kw, 1,
                wi.data(0, 0, 0, 0), c * kh * kw, matricize + i * matrix_sz,
                c * kh * kw, 0, C, (h - kh + 1) * (w - kw + 1));
    for (int j = 0; j < kc; ++j) {
      for (int k = 0; k < (h - kh + 1); ++k) {
        for (int l = 0; l < (w - kw + 1); ++l) {
          ri(i, j, k, l) += bi(j);
        }
      }
    }
  }
  delete[] matricize;
  return res;
}

py::array_t<float> relu(py::array_t<float> in) {
  auto ii = in.request();
  py::array_t<float> res(ii.shape);
  int sz = ii.size;
  float* c = static_cast<float*>(res.request().ptr);
  float* d = static_cast<float*>(ii.ptr);
  for (int i = 0; i < sz; ++i) {
    c[i] = (d[i] > 0) ? d[i] : 0;
  }
  return res;
}

py::array_t<float> maxpool2d(py::array_t<float> in, int kernel_size,
                             int strides) {
  auto ii = in.unchecked<4>();
  int n = ii.shape(0);
  int c = ii.shape(1);
  int h = ii.shape(2);
  int w = ii.shape(3);
  py::array_t<float> res(
      {n, c, (h - kernel_size) / strides + 1, (w - kernel_size) / strides + 1});
  auto ri = res.mutable_unchecked<4>();
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      for (int k = 0, o = 0; k < h; k += strides, ++o) {
        for (int l = 0, p = 0; l < w; l += strides, ++p) {
          float r = -std::numeric_limits<float>::max();
          for (int m = 0; m < kernel_size; ++m) {
            for (int n = 0; n < kernel_size; ++n) {
              r = [](float a, float b) { return (a >= b) ? a : b; }(
                      r, ii(i, j, k + m, l + n));
              // r = std::max(r, ii(i, j, k + m, l + n)); it's slower
            }
          }
          ri(i, j, o, p) = r;
        }
      }
    }
  }
  return res;
}

py::array_t<float> fc(py::array_t<float> weight, py::array_t<float> bias,
                      py::array_t<float> in) {
  auto w = weight.unchecked<2>();
  auto i = in.unchecked<2>();
  auto b = bias.unchecked<1>();
  int m = i.shape(0);
  int n = w.shape(0);
  int o = w.shape(1);
  py::array_t<float> res({m, n});
  float* C2 = static_cast<float*>(res.request().ptr);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, o, 1, i.data(0, 0),
              o, w.data(0, 0), o, 0, C2, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C2[i * n + j] += b[j];
    }
  }
  return res;
}

PYBIND11_MODULE(fp32ie, m) {
  m.def("relu", relu);
  m.def("maxpool2d", maxpool2d);
  m.def("conv2d", conv2d);
  m.def("fc", fc);
}
