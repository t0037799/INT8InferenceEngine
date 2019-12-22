#pragma once
#include "tensor.h"
void quantize(const float* M, u8_t* Q, ssize_t size, float scale,
              u8_t zp);  // asymmetric quantization
void quantize(const float* M, s8_t* Q, ssize_t size,
              float scale);  // symmetric  quantization
Tensor<u8_t>& quantize(Tensor<float>& in, float scale, u8_t zp);

void dequantize(float* M, int* Q, ssize_t size, float sa, float sb);
void dequantize(float* M, u8_t* Q, ssize_t size, float scale, u8_t zp);
Tensor<float>& dequantize(Tensor<u8_t>& in);

void down_scale(u8_t* M, int* Q, ssize_t size, float sa, float sb, float sc,
                u8_t zp_c);  // u8 = u8 * s8
