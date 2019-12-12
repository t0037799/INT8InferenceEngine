#include "pybind11/pybind11.h"

#include "layer.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "tensor.h"

namespace py = pybind11;

template <typename T>
void declare_tensor(py::module& mod) {
  py::class_<Tensor<T>>(mod, typeid(Tensor<T>).name())
      .def(py::init<>())
      .def("numpy", [](Tensor<T>& t) {
        // std::cout << "numpy\n";
        return py::array(t.shape(), t.data(), t.cap());
      });
}

PYBIND11_MODULE(tensor_core, mod) {
  mod.def("create", create_tensor);
  declare_tensor<float>(mod);
  declare_tensor<int>(mod);
  py::class_<Linear>(mod, "Linear")
      .def(py::init<py::array_t<float>, py::array_t<float>>())
      .def("__call__",
           [](Linear& layer, py::array_t<float> x) -> Tensor<float>&& {
             return std::move(layer.forward_prop(x));
           })
      .def("__call__",
           [](Linear& layer, Tensor<float>& x) -> Tensor<float>&& {
             return std::move(layer.forward_prop(std::move(x)));
           })
      .def("forward", [](Linear& layer, py::array_t<float> x) {
        return layer.forward_prop(x);
      });
}
