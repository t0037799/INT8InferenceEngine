#include "pybind11/pybind11.h"

#include "conv2d.h"
#include "layer.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "tensor.h"

namespace py = pybind11;

template <typename T>
void declare_tensor(py::module& mod) {
  py::class_<Tensor<T>>(mod, typeid(Tensor<T>).name())
      .def(py::init<>())
      .def("numpy",
           [](Tensor<T>& t) {
             // std::cout << "numpy\n";
             return py::array(t.shape(), t.data(), t.cap());
           })
      .def("reshape",
           [](Tensor<T>& t, std::vector<ssize_t>& shape) -> Tensor<T>&& {
             t.reshape(shape);
             return std::move(t);
           });
}

PYBIND11_MODULE(tensor_core, mod) {
  mod.def("create", create_tensor);
  mod.def("maxpool2d", [](Tensor<float>& in, ssize_t kernel_size, ssize_t strides) -> Tensor<float>&&{
		  return std::move(maxpool2d(std::move(in), kernel_size, strides));
		  });
  mod.def("relu", [](Tensor<float>& in) -> Tensor<float>&&{
		  return std::move(relu(std::move(in)));
		  });
  declare_tensor<float>(mod);
  declare_tensor<int>(mod);
  py::class_<Linear>(mod, "Linear")
      .def(py::init<py::array_t<float>, py::array_t<float>>())
      .def(py::init<ssize_t, ssize_t>())
      .def("load_weight",
           [](Linear& layer, py::array_t<float> w) { layer.load_weight(w); })
      .def("load_bias",
           [](Linear& layer, py::array_t<float> b) { layer.load_bias(b); })
      .def("prepare",
           [](Linear& layer) { layer.prepare(); })
      .def("convert",
           [](Linear& layer) { layer.convert(); })
      .def("__call__", [](Linear& layer, Tensor<float>& x) -> Tensor<float>&& {
        return std::move(layer.forward_prop(std::move(x)));
      });
  py::class_<Conv2d>(mod, "Conv2d")
      .def(py::init<py::array_t<float>, py::array_t<float>>())
      .def(py::init<ssize_t, ssize_t, ssize_t>())
      .def("load_weight",
           [](Conv2d& layer, py::array_t<float> w) { layer.load_weight(w); })
      .def("load_bias",
           [](Conv2d& layer, py::array_t<float> b) { layer.load_bias(b); })
      .def("prepare",
           [](Conv2d& layer) { layer.prepare(); })
      .def("convert",
           [](Conv2d& layer) { layer.convert(); })
      .def("__call__", [](Conv2d& layer, Tensor<float>& x) -> Tensor<float>&& {
        return std::move(layer.forward_prop(std::move(x)));
      });
}
