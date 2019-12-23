from .layer import Conv2d, Linear
from .module import Module
from .tensor import Tensor
import _CXX_i8ie

__all__ = [
        'tensor', 'argmax', 'relu', 'max_pool2d',
        'Linear', 'Conv2d', 'Tensor',
        'quantize', 'dequantize'
        ]


def tensor(ndarray):
    return Tensor(_CXX_i8ie.tensor(ndarray))


def argmax(x, *args, **kwargs):
    return tensor(x.numpy().argmax(*args, **kwargs))


def relu(x):
    return Tensor(_CXX_i8ie.relu(x.data))


def max_pool2d(x, kernel_size, stride):
    return Tensor(_CXX_i8ie.max_pool2d(x.data, kernel_size, stride))

def quantize(x, scale, zero_point):
    return Tensor(_CXX_i8ie.quantize(x.data, scale, zero_point))

def dequantize(x):
    return Tensor(_CXX_i8ie.dequantize(x.data))
