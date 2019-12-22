#include "layer.h"

class Conv2d : public BaseLayer {
 public:
  Conv2d() = delete;
  Conv2d(ssize_t in_channel, ssize_t out_channel, ssize_t kernel_size,
         ssize_t stride = 1, ssize_t padding = 0)
      : BaseLayer({out_channel, in_channel, kernel_size, kernel_size},
                  out_channel),
        stride_(stride),
        padding_(padding) {
    if (stride == 0) {
      throw std::exception();
    }
  }
  Conv2d(Tensor<float>& w, Tensor<float>& b) : BaseLayer(w, b) {}
  Conv2d(py::array_t<float> w, py::array_t<float> b) : BaseLayer(w, b) {}
  ~Conv2d() {}
  Tensor<float>& forward_prop(Tensor<float>&& in);
  Tensor<u8_t>& forward_prop(Tensor<u8_t>&& in);

 private:
  ssize_t padding_;
  ssize_t stride_;
};
