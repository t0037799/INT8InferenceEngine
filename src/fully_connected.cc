#include "fully_connected.h"

#include "quantize_utils.h"

Tensor<float>& Linear::forward_prop(Tensor<float>&& in) {
  ssize_t m = in.shape()[0];
  ssize_t n = weight_->shape()[0];
  ssize_t k = weight_->shape()[1];
  Tensor<float>& out = *new Tensor<float>({m, n});
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, in.data(), k,
              weight_->data(), k, 0, out.data(), n);
  for (ssize_t i = 0; i < m; ++i) {
    for (ssize_t j = 0; j < n; ++j) {
      out({i, j}) += bias_->data()[j];
    }
  }
  if (is_preparing_) {
    cal_->sample(out.data(), out.size());
  }
  return out;
}
Tensor<u8_t>& Linear::forward_prop(Tensor<u8_t>&& in) {
  ssize_t m = in.shape()[0];
  ssize_t n = q_weight_->shape()[0];
  ssize_t k = q_weight_->shape()[1];
  Tensor<u8_t>& out = *new Tensor<u8_t>({m, n});
  out.scale() = scale_;
  out.zero_point() = zero_point_;
  int* C = new int[m * n];
  int* oc = new int[n];
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {  // calculate offset after mul
    float t = 0;
    for (int j = 0; j < k; ++j) {
      t += in.zero_point() * (*q_weight_)({i, j});
    }
    oc[i] = -t;
  }
  cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasRowOffset, m,
                     n, k, 1, in.data(), k, 0, q_weight_->data(), k, 0, 0, C, n,
                     oc);
  for (ssize_t i = 0; i < m; ++i) {
    for (ssize_t j = 0; j < n; ++j) {
      C[i * n + j] += q_bias_->data()[j] / in.scale();
    }
  }
  down_scale(out.data(), C, out.size(), in.scale(), q_weight_->scale(),
             out.scale(), out.zero_point());
  delete[] C;
  delete[] oc;
  return out;
}

void declare_linear(py::module& mod) {
  py::class_<Linear>(mod, "Linear")
      .def(py::init<py::array_t<float>, py::array_t<float>>())
      .def(py::init<Tensor<float>&, Tensor<float>&>())
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
}
