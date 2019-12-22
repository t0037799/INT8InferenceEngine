#pragma once

#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

using u8_t = unsigned char;
using s8_t = char;

template <typename T>
class Tensor {
 public:
  Tensor() {}
  Tensor(Tensor<T> const& t)
      : size_(t.size_),
        shape_(t.shape_),
        is_quantized_(t.is_quantized_),
        scale_(t.scale_),
        zero_point_(t.zero_point_) {
    data_ = new T[t.size_];
    std::memcpy(data_, t.data_, t.size_ * sizeof(T));
    cap_ = py::capsule(data_, [](void* f) { delete[] static_cast<T*>(f); });
  }
  Tensor(Tensor<T>&& t)
      : size_(t.size_),
        shape_(t.shape_),
        is_quantized_(t.is_quantized_),
        scale_(t.scale_),
        zero_point_(t.zero_point_) {
    data_ = t.data_;
    cap_ = t.cap_;
    t.data_ = nullptr;
  }
  Tensor(py::array_t<float> ndarray) {
    auto info = ndarray.request();
    shape_ = info.shape;
    size_ = info.size;
    data_ = new float[info.size];
    std::memcpy(data_, info.ptr, info.size * sizeof(float));
    cap_ = py::capsule(data_, [](void* f) { delete[] static_cast<float*>(f); });
  }
  Tensor(ssize_t size) {
    shape_ = std::vector<ssize_t>{size};
    size_ = size;
    data_ = new T[size_];
    cap_ = py::capsule(data_, [](void* f) { delete[] static_cast<T*>(f); });
  }
  Tensor(std::vector<ssize_t> shape) {
    shape_ = shape;
    size_ = 1;
    for_each(shape_.begin(), shape_.end(),
             [this](ssize_t n) { this->size_ *= n; });
    data_ = new T[size_];
    cap_ = py::capsule(data_, [](void* f) { delete[] static_cast<T*>(f); });
  }
  ~Tensor() {}
  T& operator()(ssize_t m, ssize_t n) {
    auto [i, j] = std::make_tuple(shape_[0], shape_[1]);
    return *(data_ + m * j + n);
  }
  const T& operator()(ssize_t m, ssize_t n) const {
    auto [i, j] = std::make_tuple(shape_[0], shape_[1]);
    return *(data_ + m * j + n);
  }
  T& operator()(ssize_t m, ssize_t n, ssize_t o, ssize_t p) {
    auto [h, i, j, k] =
        std::make_tuple(shape_[0], shape_[1], shape_[2], shape_[3]);
    return *(data_ + m * i * j * k + n * j * k + o * k + p);
  }
  const T& operator()(ssize_t m, ssize_t n, ssize_t o, ssize_t p) const {
    auto [h, i, j, k] =
        std::make_tuple(shape_[0], shape_[1], shape_[2], shape_[3]);
    return *(data_ + m * i * j * k + n * j * k + o * k + p);
  }
  void load_numpy(py::array_t<float> ndarray) {
    auto info = ndarray.request();
    if (shape_ != info.shape) {
      size_ = info.size;
      shape_ = info.shape;
      data_ = new float[info.size];
      cap_ = py::capsule(data_, [](void* f) {
        delete[] static_cast<T*>(f);
      });  // old capsule will release old data_
    }
    std::memcpy(data_, info.ptr, info.size * sizeof(float));
  }

  Tensor& tensor_view(std::vector<ssize_t> shape) {
    Tensor<T>& out = *new Tensor<T>();
    out.shape_ = shape;
    out.size_ = size_;
    out.data_ = data_;
    out.cap_ = cap_;
    cap_.dec_ref();
    out.is_quantized_ = is_quantized_;
    out.scale_ = scale_;
    out.zero_point_ = zero_point_;
    return out;
  }

  Tensor<T>& reshape(std::vector<ssize_t>& shape) {
    ssize_t midx = -1;
    ssize_t sz = 1;
    for (ssize_t i = 0; i < shape.size(); ++i) {
      if (shape[i] < 0) {
        if (midx == -1) {
          midx = i;
        } else {
          throw std::exception();
        }
      } else if (shape[i] == 0) {
        throw std::exception();
      } else {
        sz *= shape[i];
      }
    }
    if (midx >= 0) {
      if (size_ % sz != 0) {
        throw std::exception();
      }
      shape[midx] = size_ / sz;
      sz *= shape[midx];
    }
    if (sz != size_) {
      throw std::exception();
    }
    return tensor_view(shape);
  }
  const ssize_t size() const { return size_; }
  const std::vector<ssize_t>& shape() const { return shape_; };
  const py::capsule& cap() const { return cap_; }
  T* data() { return data_; }

  const bool is_quantized() const { return is_quantized_; }
  const float scale() const { return scale_; }
  float& scale() { return scale_; }
  const u8_t zero_point() const { return zero_point_; }
  u8_t& zero_point() { return zero_point_; }

 private:
  // generic
  py::capsule cap_;
  T* data_;
  ssize_t size_;
  std::vector<ssize_t> shape_;
  // quantization
  bool is_quantized_ = false;
  float scale_;
  u8_t zero_point_;
};
