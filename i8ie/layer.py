from .tensor import Tensor
import _CXX_i8ie


class Layer:
    def __call__(self, x):
        return Tensor(self.layer(x.data))

    def load_weight(self, weight):
        self.layer.load_weight(weight)

    def load_bias(self, bias):
        self.layer.load_bias(bias)

    def prepare(self):
        self.layer.prepare()

    def convert(self):
        self.layer.convert()


class Linear(Layer):
    def __init__(self, in_channels, out_channels):
        self.layer = _CXX_i8ie.Linear(in_channels, out_channels)


class Conv2d(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        self.layer = _CXX_i8ie.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding)
