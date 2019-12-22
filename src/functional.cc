#include "functional.h"

#include <limits>

template <typename T>
Tensor<T>& relu(Tensor<T>&& in) {
  Tensor<T>& out = *new Tensor<T>(in.shape());
#pragma omp parallel for
  for (ssize_t i = 0; i < in.size(); ++i) {
    out.data()[i] = (in.data()[i] > 0) ? in.data()[i] : 0;
  }
  return out;
}

template <>
Tensor<u8_t>& relu<u8_t>(Tensor<u8_t>&& in) {
  Tensor<u8_t>& out = *new Tensor<u8_t>(in.shape());
  out.scale() = in.scale();
  out.zero_point() = in.zero_point();
#pragma omp parallel for
  for (ssize_t i = 0; i < in.size(); ++i) {
    out.data()[i] =
        (in.data()[i] > in.zero_point()) ? in.data()[i] : in.zero_point();
  }
  return out;
}

template <typename T>
T min() {
  return -std::numeric_limits<T>::max();
}
template <>
u8_t min<u8_t>() {
  return 0;
}
template <typename T>
Tensor<T>& max_pool2d(Tensor<T>&& in, ssize_t kernel_size, ssize_t strides) {
  auto shape = in.shape();
  auto [n, c, h, w] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
  ssize_t oh = (h - kernel_size) / strides + 1;
  ssize_t ow = (w - kernel_size) / strides + 1;
  std::cerr << n << c << oh << h << kernel_size << "he\n";
  Tensor<T>& out = *new Tensor<T>({n, c, oh, ow});
  out.scale() = in.scale();
  out.zero_point() = in.zero_point();
#pragma omp parallel for
  for (ssize_t i = 0; i < n; ++i) {
    for (ssize_t j = 0; j < c; ++j) {
      for (ssize_t k = 0, o = 0; o < oh; k += strides, ++o) {
        for (ssize_t l = 0, p = 0; p < ow; l += strides, ++p) {
          T max = min<T>();
          for (ssize_t m = 0; m < kernel_size; ++m) {
            for (ssize_t n = 0; n < kernel_size; ++n) {
              max = [](T a, T b) { return (a >= b) ? a : b; }(
                        max, in(i, j, k + m, l + n));
              // r = std::max(r, in(i, j, k + m, l + n)); it's slower
            }
          }
          out(i, j, o, p) = max;
        }
      }
    }
  }
  return out;
}

template <typename T>
void declare_tensor_func(py::module& mod) {
  mod.def(
      "max_pool2d",
      [](Tensor<T>& in, ssize_t kernel_size, ssize_t strides) -> Tensor<T>&& {
        return std::move(max_pool2d(std::move(in), kernel_size, strides));
      });
  mod.def("relu", [](Tensor<T>& in) -> Tensor<T>&& {
    return std::move(relu(std::move(in)));
  });
}

void declare_tensor_funcs(py::module& mod) {
  declare_tensor_func<float>(mod);
  declare_tensor_func<u8_t>(mod);
  declare_tensor_func<s8_t>(mod);
}
