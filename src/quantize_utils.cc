#include "quantize_utils.h"

void quantize(const float* M, u8_t* Q, ssize_t size, float scale,
              u8_t zp) {  // asymmetric quantization
  for (ssize_t i = 0; i < size; ++i) {
    float t = M[i] / scale + zp;
    Q[i] = (t >= 255) ? 255 : (t < 0) ? 0 : t;  // prevent exceed bound
                                                // Q[i] =  t ;
  }
}

void quantize(const float* M, s8_t* Q, ssize_t size,
              float scale) {  // symmetric  quantization
  for (ssize_t i = 0; i < size; ++i) {
    float t = M[i] / scale;
    Q[i] = (t <= -127) ? -127 : (t >= 127) ? 127 : t;  // prevent exceed bound
                                                       // Q[i] =  t ;
  }
}

void dequantize(float* M, int* Q, ssize_t size, float sa, float sb) {
  for (ssize_t i = 0; i < size; ++i) {
    M[i] = Q[i] * sa * sb;
  }
}

void down_scale(u8_t* M, int* Q, ssize_t size, float sa, float sb, float sc,
                u8_t zp_c) {  // u8 = u8 * s8
  for (ssize_t i = 0; i < size; ++i) {
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

Tensor<u8_t>& quantize(Tensor<float>& in, float scale, u8_t zp) {
  Tensor<u8_t>& out = *new Tensor<u8_t>(in.shape());
  out.scale() = scale;
  out.zero_point() = zp;
  for (ssize_t i = 0; i < out.size(); ++i) {
    out.data()[i] = in.data()[i] / scale + zp;
  }
  return out;
}

Tensor<float>& dequantize(Tensor<u8_t>& in) {
  Tensor<float>& out = *new Tensor<float>(in.shape());
  dequantize(out.data(), in.data(), in.size(), in.scale(), in.zero_point());
  return out;
}
