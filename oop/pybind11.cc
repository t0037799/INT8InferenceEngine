#include "pybind11/pybind11.h"

#include "conv2d.h"
#include "layer.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "quant_utils.h"
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
      .def("zp", [](Tensor<T>& t) { return t.zero_point(); })
      .def("sc", [](Tensor<T>& t) { return t.scale(); })
      .def("sum",
           [](Tensor<T>& t) {
             float s = 0;
             for (ssize_t i = 0; i < t.size(); ++i) {
               s += t.data()[i];
             }
             return s;
           })
      .def("reshape",
           [](Tensor<T>& t, std::vector<ssize_t>& shape) -> Tensor<T>&& {
             t.reshape(shape);
             return std::move(t);
           });
}

template <typename T>
void declare_tensor_func(py::module& mod) {
  mod.def(
      "maxpool2d",
      [](Tensor<T>& in, ssize_t kernel_size, ssize_t strides) -> Tensor<T>&& {
        return std::move(maxpool2d(std::move(in), kernel_size, strides));
      });
  mod.def("relu", [](Tensor<T>& in) -> Tensor<T>&& {
    return std::move(relu(std::move(in)));
  });
}

PYBIND11_MODULE(tensor_core, mod) {
  mod.def("create", create_tensor);
  mod.def("quantize",
          [](Tensor<float>& in, float scale, u8_t zp) -> Tensor<u8_t>&& {
            return std::move(quantt(in, scale, zp));
          });
  mod.def("dequantize", [](Tensor<u8_t>& in) -> Tensor<float>&& {
    return std::move(dequantt(in));
  });
  declare_tensor<float>(mod);
  declare_tensor<u8_t>(mod);
  declare_tensor_func<float>(mod);
  declare_tensor_func<u8_t>(mod);
  py::class_<Linear>(mod, "Linear")
      .def(py::init<py::array_t<float>, py::array_t<float>>())
      .def(py::init<ssize_t, ssize_t>())
      .def("load_weight",
           [](Linear& layer, py::array_t<float> w) { layer.load_weight(w); })
      .def("load_bias",
           [](Linear& layer, py::array_t<float> b) { layer.load_bias(b); })
      .def("prepare", [](Linear& layer) { layer.prepare(); })
      .def("convert", [](Linear& layer) { layer.convert(); })
      .def("__call__",
           [](Linear& layer, Tensor<float>& x) -> Tensor<float>&& {
             return std::move(layer.forward_prop(std::move(x)));
           })
      .def("__call__", [](Linear& layer, Tensor<u8_t>& x) -> Tensor<u8_t>&& {
        return std::move(layer.forward_prop(std::move(x)));
      });
  py::class_<Conv2d>(mod, "Conv2d")
      .def(py::init<py::array_t<float>, py::array_t<float>>())
      .def(py::init<ssize_t, ssize_t, ssize_t>())
      .def("load_weight",
           [](Conv2d& layer, py::array_t<float> w) { layer.load_weight(w); })
      .def("load_bias",
           [](Conv2d& layer, py::array_t<float> b) { layer.load_bias(b); })
      .def("prepare", [](Conv2d& layer) { layer.prepare(); })
      .def("convert", [](Conv2d& layer) { layer.convert(); })
      .def("__call__",
           [](Conv2d& layer, Tensor<float>& x) -> Tensor<float>&& {
             return std::move(layer.forward_prop(std::move(x)));
           })
      .def("__call__", [](Conv2d& layer, Tensor<u8_t>& x) -> Tensor<u8_t>&& {
        return std::move(layer.forward_prop(std::move(x)));
      });
}
