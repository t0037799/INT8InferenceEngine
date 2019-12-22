#include "conv2d.h"

#include "quantize_utils.h"

template <typename T>
static void im2col_tile(T* M, const T* I, int c, int h, int w, int kh, int kw,
                        int i, int j) {
  for (int m = 0; m < kw; ++m) {
    for (int l = 0; l < kh; ++l) {
      for (int k = 0; k < c; ++k) {
        M[k * kh * kw + l * kw + m] = I[(h * w) * k + (i + l) * w + (j + m)];
      }
    }
  }
}

template <typename T>
static void im2col_tile(T* M, const T* I, int c, int h, int w, int kh, int kw,
                        int i, int j, int zero_point) {
  for (int m = 0; m < kw; ++m) {
    for (int l = 0; l < kh; ++l) {
      for (int k = 0; k < c; ++k) {
        const ssize_t idx = (h * w) * k + (i + l) * w + (j + m);
        if ((i + l) < 0 || (j + m) < 0 || (i + l) >= h || (j + m) >= w) {
          M[k * kh * kw + l * kw + m] = zero_point;
        } else {
          M[k * kh * kw + l * kw + m] = I[idx];
        }
      }
    }
  }
}

template <typename T>
static void im2col(T* M, const T* I, int c, int h, int w, int kh, int kw,
                   ssize_t stride, ssize_t padding, int zero_point) {
  ssize_t oh = (h - kh + 2 * padding) / stride + 1;
  ssize_t ow = (w - kw + 2 * padding) / stride + 1;
  for (int i = -padding, ti = 0; ti < oh; i += stride, ++ti) {
    for (int j = -padding, tj = 0; tj < ow; j += stride, ++tj) {
      if (padding != 0) {
        im2col_tile(M + (ti * ow + tj) * c * kh * kw, I, c, h, w, kh, kw, i, j,
                    zero_point);
      } else {
        im2col_tile(M + (ti * ow + tj) * c * kh * kw, I, c, h, w, kh, kw, i, j);
      }
    }
  }
}

template <typename T>
static void transpose(T* data, ssize_t nrow, ssize_t ncol) {
  T* back_data = new T[nrow * ncol];
  std::memcpy(back_data, data, nrow * ncol * sizeof(T));
  for (ssize_t i = 0; i < nrow; ++i) {
    for (ssize_t j = 0; j < ncol; ++j) {
      data[j * nrow + i] = back_data[i * ncol + j];
    }
  }
  delete[] back_data;
}

Tensor<float>& Conv2d::forward_prop(Tensor<float>&& in) {
  ssize_t n = in.shape()[0];
  ssize_t c = in.shape()[1];
  ssize_t h = in.shape()[2];
  ssize_t w = in.shape()[3];
  ssize_t kc = weight_->shape()[0];
  ssize_t kh = weight_->shape()[2];
  ssize_t kw = weight_->shape()[3];
  ssize_t oh = (h - kh + 2 * padding_) / stride_ + 1;
  ssize_t ow = (w - kw + 2 * padding_) / stride_ + 1;
  Tensor<float>* outp = new Tensor<float>({n, kc, oh, ow});
  Tensor<float>& out = *outp;
  ssize_t mat_m = kc;
  ssize_t mat_n = oh * ow;
  ssize_t mat_k = c * kh * kw;
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    float* matricize = new float[mat_k * mat_n];
    im2col(matricize, &in(i, 0, 0, 0), c, h, w, kh, kw, stride_, padding_, 0);
    float* C = const_cast<float*>(&out(i, 0, 0, 0));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mat_m, mat_n, mat_k, 1,
                weight_->data(), mat_k, matricize, mat_k, 0, C, mat_n);
    for (int j = 0; j < kc; ++j) {
      for (int k = 0; k < oh; ++k) {
        for (int l = 0; l < ow; ++l) {
          out(i, j, k, l) += bias_->data()[j];
        }
      }
    }
    delete[] matricize;
  }
  if (is_preparing_) {
    cal_->sample(out.data(), out.size());
  }
  return out;
}

Tensor<u8_t>& Conv2d::forward_prop(Tensor<u8_t>&& in) {
  ssize_t n = in.shape()[0];
  ssize_t c = in.shape()[1];
  ssize_t h = in.shape()[2];
  ssize_t w = in.shape()[3];
  ssize_t kc = q_weight_->shape()[0];
  ssize_t kh = q_weight_->shape()[2];
  ssize_t kw = q_weight_->shape()[3];
  ssize_t oh = (h - kh + 2 * padding_) / stride_ + 1;
  ssize_t ow = (w - kw + 2 * padding_) / stride_ + 1;
  Tensor<u8_t>* outp = new Tensor<u8_t>({n, kc, oh, ow});
  Tensor<u8_t>& out = *outp;
  out.scale() = scale_;
  out.zero_point() = zero_point_;
  ssize_t mat_m = oh * ow;
  ssize_t mat_n = kc;
  ssize_t mat_k = c * kh * kw;
  int* oc = new int[mat_n];
  for (int j = 0; j < mat_n; ++j) {  // calculate offset after mul
    float t = 0;
    for (int k = 0; k < mat_k; ++k) {
      t += in.zero_point() * q_weight_->data()[j * mat_k + k];
    }
    oc[j] = q_bias_->data()[j] / in.scale() - t;  // bias count in offset
  }
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    u8_t* matricize = new u8_t[mat_m * mat_k];
    int* C = new int[mat_m * mat_n];
    im2col(matricize, &in(i, 0, 0, 0), c, h, w, kh, kw, stride_, padding_,
           in.zero_point());
    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasRowOffset,
                       mat_m, mat_n, mat_k, 1, matricize, mat_k, 0,
                       q_weight_->data(), mat_k, 0, 0, C, mat_n, oc);
    down_scale(&out(i, 0, 0, 0), C, mat_m * mat_n, in.scale(),
               q_weight_->scale(), out.scale(), out.zero_point());
    transpose(&out(i, 0, 0, 0), mat_m, mat_n);
    delete[] C;
    delete[] matricize;
  }
  delete[] oc;
  return out;
}

void declare_conv2d(py::module& mod) {
  py::class_<Conv2d>(mod, "Conv2d")
      .def(py::init<py::array_t<float>, py::array_t<float>>())
      .def(py::init<Tensor<float>&, Tensor<float>&>())
      .def(py::init<ssize_t, ssize_t, ssize_t, ssize_t, ssize_t>(),
           py::arg("in_channels"), py::arg("out_channels"),
           py::arg("kernel_size"), py::arg("stride") = 1,
           py::arg("padding") = 0)
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
