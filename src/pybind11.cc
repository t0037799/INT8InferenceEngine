#include "pybind11/pybind11.h"

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "quantize_utils.h"
#include "tensor.h"

namespace py = pybind11;

template <typename T>
void declare_tensor(py::module& mod) {
  py::class_<Tensor<T>>(mod, typeid(Tensor<T>).name())
      .def(py::init<>())
      .def("numpy",
           [](Tensor<T>& t) { return py::array(t.shape(), t.data(), t.cap()); })
      .def("zero_point", [](Tensor<T>& t) { return t.zero_point(); })
      .def("scale", [](Tensor<T>& t) { return t.scale(); })
      .def("sum",
           [](Tensor<T>& t) {
             float s = 0;
             for (ssize_t i = 0; i < t.size(); ++i) {
               s += t.data()[i];
             }
             return s;
           })
      .def("ref_count", [](Tensor<T>& t) { return t.cap().ref_count(); })
      .def("reshape",
           [](Tensor<T>& t, std::vector<ssize_t>& shape) -> Tensor<T>&& {
             return std::move(t.reshape(shape));
           });
}

void declare_tensor_funcs(py::module&);
void declare_linear(py::module&);
void declare_conv2d(py::module&);

PYBIND11_MODULE(_CXX_i8ie, mod) {
  mod.def("tensor", [](py::array_t<float> ndarray) {
    return std::unique_ptr<Tensor<float>>(new Tensor<float>(ndarray));
  });
  mod.def(
      "quantize",
      [](Tensor<float>& in, float scale, u8_t zero_point) -> Tensor<u8_t>&& {
        return std::move(quantize(in, scale, zero_point));
      });
  mod.def("dequantize", [](Tensor<u8_t>& in) -> Tensor<float>&& {
    return std::move(dequantize(in));
  });
  declare_tensor<float>(mod);
  declare_tensor<u8_t>(mod);
  declare_tensor<s8_t>(mod);
  declare_tensor_funcs(mod);
  declare_linear(mod);
  declare_conv2d(mod);
}
