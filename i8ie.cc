#include <iostream>
#include <memory>

#include "mkl.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

using u8_t = unsigned char;
using s8_t = char;

void quantize(const float* M, u8_t* Q, int m, int n, float scale,
              u8_t zp) {  // asymmetric quantization
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float t = M[i * n + j] / scale + zp;
      Q[i * n + j] =
          (t >= 255) ? 255 : (t < 0) ? 0 : t;  // prevent exceed bound
      // Q[i*n+j] =  t ;
    }
  }
}

void quantize(const float* M, s8_t* Q, int m, int n,
              float scale) {  // symmetric  quantization
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float t = M[i * n + j] / scale;
      Q[i * n + j] =
          (t <= -127) ? -127 : (t >= 127) ? 127 : t;  // prevent exceed bound
      // Q[i*n+j] =  t ;
    }
  }
}

void dequantize(float* M, int* Q, int m, int n, float sa, float sb) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      M[i * n + j] = Q[i * n + j] * sa * sb;
    }
  }
}

py::array_t<float> qfc(py::array_t<float> weight, py::array_t<float> bias,
                       py::array_t<float> in) {
  auto w = weight.unchecked<2>();
  auto i = in.unchecked<2>();
  auto b = bias.unchecked<1>();
  int m = i.shape(0);
  int n = w.shape(0);
  int o = w.shape(1);
  py::array_t<float> res({m, n});
  float* C2 = static_cast<float*>(res.request().ptr);
  u8_t* A = new u8_t[m * o];
  s8_t* B = new s8_t[o * n];
  int* C = new int[m * n];
  int* oc = new int[n];
  float sa = 0.02;  // hard coded hyper params
  float sb = 0.009;
  u8_t zp = 50;
  for (int i = 0; i < n; ++i) {  // calculate offset after mul
    float t = 0;
    for (int j = 0; j < o; ++j) {
      t += zp * w(i, j) / sb;
    }
    oc[i] = -t;
  }

  quantize(i.data(0, 0), A, m, o, sa, zp);
  quantize(w.data(0, 0), B, n, o, sb);
  cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasRowOffset, m,
                     n, o, 1, A, o, 0, B, o, 0, 0, C, n, oc);
  dequantize(C2, C, m, n, sa, sb);
  for (int i = 0; i < m; ++i) {  // add bias
    for (int j = 0; j < n; ++j) {
      C2[i * n + j] += b[j];
    }
  }
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] oc;
  return res;
}

PYBIND11_MODULE(i8ie, m) { m.def("qfc", qfc); }
