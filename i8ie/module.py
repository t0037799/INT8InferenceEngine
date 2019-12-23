from .tensor import Tensor
from .layer import Layer
import _CXX_i8ie


class Module:
    def __init__(self):
        self.is_quant = False

    def load(self, state_dict):
        for key in state_dict:
            name, attr = key.split('.')
            if attr == 'weight':
                self.__dict__[name].load_weight(state_dict[key])
            elif attr == 'bias':
                self.__dict__[name].load_bias(state_dict[key])

    def __call__(self, x):
        if self.is_quant:
            x = Tensor(_CXX_i8ie.quantize(x.data, 0.025, 127))
        x = self.forward(x)
        if self.is_quant:
            x = Tensor(_CXX_i8ie.dequantize(x.data))
        return x

    def prepare(self):
        for attr, val in self.__dict__.items():
            if issubclass(type(val), Layer):
                val.prepare()

    def convert(self):
        for attr,val in self.__dict__.items():
            if issubclass(type(val), Layer):
                val.convert()
        self.is_quant = True
