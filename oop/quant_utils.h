#include "tensor.h"
#pragma once

using u8_t = unsigned char;
using s8_t = char;

void quantize(const float* M, u8_t* Q, ssize_t size, float scale,
              u8_t zp) {  // asymmetric quantization
  for (int i = 0; i < size; ++i) {
    float t = M[i] / scale + zp;
    Q[i] = (t >= 255) ? 255 : (t < 0) ? 0 : t;  // prevent exceed bound
                                                // Q[i] =  t ;
  }
}

void quantize(const float* M, s8_t* Q, ssize_t size,
              float scale) {  // symmetric  quantization
  for (int i = 0; i < size; ++i) {
    float t = M[i] / scale;
    Q[i] = (t <= -127) ? -127 : (t >= 127) ? 127 : t;  // prevent exceed bound
                                                       // Q[i] =  t ;
  }
}

void dequantize(float* M, int* Q, ssize_t size, float sa, float sb) {
  for (int i = 0; i < size; ++i) {
    M[i] = Q[i] * sa * sb;
  }
}

void down_scale(u8_t* M, int* Q, ssize_t size, float sa, float sb, float sc,
                u8_t zp_c) {  // u8 = u8 * s8
  for (int i = 0; i < size; ++i) {
    float dequant = Q[i] * sa * sb;
    float quant = dequant / sc + zp_c;
    M[i] =
        (quant >= 255) ? 255 : (quant < 0) ? 0 : quant;  // prevent exceed bound
                                                         // M[i*n+j] =  quant ;
  }
}

void dequantize(float* M, u8_t* Q, ssize_t size, float scale, u8_t zp) {
  for (ssize_t i = 0; i < size; ++i) {
    M[i] = (Q[i] - zp) * scale;
  }
}

Tensor<u8_t>& quantt(Tensor<float>& in, float scale, u8_t zp) {
  Tensor<u8_t>* outp = new Tensor<u8_t>(in.shape());
  outp->scale() = scale;
  outp->zero_point() = zp;
  for (ssize_t i = 0; i < outp->size(); ++i) {
    outp->data()[i] = in.data()[i] / scale + zp;
  }
  return *outp;
}

Tensor<float>& dequantt(Tensor<u8_t>& in) {
  Tensor<float>* outp = new Tensor<float>(in.shape());
  dequantize(outp->data(), in.data(), in.size(), in.scale(), in.zero_point());
  return *outp;
}
