#pragma once
#include "tensor.h"

template <typename T>
Tensor<T>& relu(Tensor<T>&& in);
template <typename T>
Tensor<T>& max_pool2d(Tensor<T>&& in, ssize_t kernel_size, ssize_t strides);
