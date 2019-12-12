#pragma once

#include <chrono>
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
  enum qtype_t {
    non_quantized,
    asym_per_tensor,
    asym_per_channel,
    sym_per_tensor,
    sym_per_channel
  };
  enum dtype_t { tensor_fp32, tensor_int32, tensor_uint8, tensor_sint8 };

 public:
  Tensor() { std::cout << "default ctor\n"; }
  Tensor(Tensor<T> const& t) {
    std::cout << "copy ctor\n";
    this->size_ = t.size_;
    this->ndim_ = t.ndim_;
    this->shape_ = t.shape_;
    data_ = new T[t.size_];
    memcpy(data_, t.data_, t.size_ * sizeof(T));
    cap_ = py::capsule(data_, [](void* f) { delete[] static_cast<T*>(f); });
  }
  Tensor(Tensor<T>&& t) {
    // std::cout << "move ctor\n";
    /*
      std::cout << "move ctor";
      for(auto i : t.shape_){
              std::cout << i << " ";
      }
      std::cout << "\n";
      */
    this->size_ = t.size_;
    this->ndim_ = t.ndim_;
    this->shape_ = std::move(t.shape_);
    this->data_ = t.data_;
    t.data_ = nullptr;
    cap_ = py::capsule(data_, [](void* f) {
      // std::cerr << "release data_\n";
      delete[] static_cast<T*>(f);
    });
  }
  Tensor& operator=(Tensor const& t) { std::cout << "copy assign\n"; }
  Tensor& operator=(Tensor&& t) { std::cout << "move assign\n"; }
  Tensor(py::array_t<float> ndarray) {
    auto info = ndarray.request();
    shape_ = std::move(info.shape);
    size_ = std::move(info.size);
    ndim_ = info.ndim;
    data_ = new float[info.size];
    std::memcpy(data_, info.ptr, info.size * sizeof(float));
    cap_ = py::capsule(data_, [](void* f) {
      // std::cerr << "release data_\n";
      delete[] static_cast<T*>(f);
    });
    // std::cout << "numpy ctor\n";
    /*
      std::cout << "numpy ctor";
      for(auto i : shape_){
              std::cout << i << " ";
      }
      std::cout << "\n";
      */
  }
  Tensor(ssize_t size) {
    // std::cout << "size ctop\n";
    shape_ = std::vector<ssize_t>{size};
    size_ = size;
    ndim_ = 1;
    data_ = new float[size_];
    cap_ = py::capsule(data_, [](void* f) {
      // std::cerr << "release data_\n";
      delete[] static_cast<T*>(f);
    });
  }
  Tensor(std::vector<ssize_t> shape) {
    // std::cout << "shape ctop\n";
    /*
    std::cout << "shape ctor";
    for(auto i : shape){
            std::cout << i << " ";
    }
    std::cout << "\n";
    */
    shape_ = std::move(shape);
    size_ = 1;
    for_each(shape_.begin(), shape_.end(),
             [this](ssize_t n) { this->size_ *= n; });
    ndim_ = shape_.size();
    data_ = new float[size_];
    cap_ = py::capsule(data_, [](void* f) {
      // std::cerr << "release data_\n";
      delete[] static_cast<T*>(f);
    });
  }
  ~Tensor() {
    // std::cout << "bye c++\n";
    /*
	std::cout << "bye c++";
      for(auto i : shape_){
              std::cout << i << " ";
      }
      std::cout << "\n";
      */
  }
  void load_numpy(py::array_t<float> ndarray) {
    auto info = ndarray.request();
    if (shape_ != info.shape) {
      if (data_ != nullptr) {
        delete[] data_;
      }
      shape_ = std::move(info.shape);
      size_ = info.size;
      ndim_ = info.ndim;
      data_ = new float[info.size];
    }
    std::memcpy(data_, info.ptr, info.size * sizeof(float));
  }
  const bool is_quantized() const { return is_quantized_; }
  const int quantize_type() const { return quantize_type_; }
  const float scale() const { return scale_; }
  const float zero_point() const { return zero_point_; }
  const std::vector<ssize_t>& shape() const { return shape_; };
  T* data() { return data_; }
  const py::capsule& cap() const { return cap_; }

 private:
  // generic
  py::capsule cap_;
  T* data_;
  ssize_t size_;
  ssize_t ndim_;
  std::vector<ssize_t> shape_;
  dtype_t dtype_;
  // quantization
  bool is_quantized_;
  qtype_t quantize_type_;
  float scale_;
  u8_t zero_point_;
};

class PyTensor {
};

std::unique_ptr<Tensor<float>> create_tensor(py::array_t<float> ndarray) {
  return std::unique_ptr<Tensor<float>>(new Tensor<float>(ndarray));
}