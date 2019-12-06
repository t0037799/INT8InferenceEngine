#include <iostream>
#include <memory>

#include "mkl.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

using u8_t = unsigned char;
using s8_t = char;

void quantize(float* M, u8_t* Q, int m, int n, float scale,
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

void quantize(float* M, s8_t* Q, int m, int n,
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

py::array_t<float> qfc(py::buffer weight, py::buffer bias, py::buffer in) {
  py::buffer_info w_info = weight.request();
  py::buffer_info b_info = bias.request();
  py::buffer_info i_info = in.request();
  int m = i_info.shape[0];
  int n = w_info.shape[0];
  int o = w_info.shape[1];
  std::shared_ptr<float> C2(new float[m * n]());
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
      t += zp * ((float*)(w_info.ptr))[i * o + j] / sb;
    }
    oc[i] = -t;
  }

  quantize((float*)i_info.ptr, A, m, o, sa, zp);
  quantize((float*)w_info.ptr, B, n, o, sb);
  cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasRowOffset, m,
                     n, o, 1, A, o, 0, B, o, 0, 0, C, n, oc);
  dequantize(C2.get(), C, m, n, sa, sb);
  for (int i = 0; i < m; ++i) {  // add bias
    for (int j = 0; j < n; ++j) {
      C2.get()[i * n + j] += ((float*)b_info.ptr)[j];
    }
  }
  return py::array_t<float>(py::buffer_info(
      C2.get(), sizeof(float), py::format_descriptor<float>::format(), 2,
      {m, n}, {sizeof(float) * n, sizeof(float)}));
}

py::array_t<float> fc(py::buffer weight, py::buffer bias, py::buffer in) {
  py::buffer_info w_info = weight.request();
  py::buffer_info b_info = bias.request();
  py::buffer_info i_info = in.request();
  int m = i_info.shape[0];
  int n = w_info.shape[0];
  int o = w_info.shape[1];
  std::shared_ptr<float> C2(new float[m * n]);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, o, 1,
              (float*)i_info.ptr, o, (float*)w_info.ptr, o, 0, C2.get(), n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C2.get()[i * n + j] += ((float*)b_info.ptr)[j];
    }
  }
  return py::array_t<float>(py::buffer_info(
      C2.get(), sizeof(float), py::format_descriptor<float>::format(), 2,
      {m, n}, {sizeof(float) * n, sizeof(float)}));
}

PYBIND11_MODULE(i8ie, m) {
  m.def("fc", fc);
  m.def("qfc", qfc);
}
