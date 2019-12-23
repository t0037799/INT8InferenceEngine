from .layer import Conv2d, Linear
from .module import Module
from .tensor import Tensor
import _CXX_i8ie

__all__ = [
        'tensor', 'argmax', 'relu', 'max_pool2d',
        'Linear', 'Conv2d', 'Tensor'
        ]


def tensor(ndarray):
    return Tensor(_CXX_i8ie.tensor(ndarray))


def argmax(x, *args, **kwargs):
    return tensor(x.numpy().argmax(*args, **kwargs))


def relu(x):
    return Tensor(_CXX_i8ie.relu(x.data))


def max_pool2d(x, kernel_size, stride):
    return Tensor(_CXX_i8ie.max_pool2d(x.data, kernel_size, stride))
