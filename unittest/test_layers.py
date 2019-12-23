import unittest
import i8ie
import numpy as np
import torch

class TestTensorLayers(unittest.TestCase):
    def get_ndarray(self, shape):
        return np.random.uniform(-1,1,shape).astype(np.float32)

    def assertEqualArray(self, a, b):
        return self.assertTrue(np.allclose(a,b, atol=0.1))

    def test_linear(self):
        ifc = i8ie.Linear(800, 500)
        tfc = torch.nn.Linear(800, 500)

        weight = self.get_ndarray((500,800))
        bias = self.get_ndarray(500)

        tfc.weight.data= torch.tensor(weight)
        tfc.bias.data= torch.tensor(bias)
        ifc.load_weight(weight)
        ifc.load_bias(bias)

        x = self.get_ndarray((200,800))
        self.assertEqualArray(ifc(i8ie.tensor(x)).numpy(), tfc(torch.tensor(x)).detach().numpy())

    def test_conv2d1(self):
        iconv = i8ie.Conv2d(10, 20, 3)
        tconv = torch.nn.Conv2d(10, 20, 3)

        weight = self.get_ndarray((20,10,3,3))
        bias = self.get_ndarray(20)

        tconv.weight.data= torch.tensor(weight)
        tconv.bias.data= torch.tensor(bias)
        iconv.load_weight(weight)
        iconv.load_bias(bias)

        x = self.get_ndarray((30,10, 22, 22))
        self.assertEqualArray(iconv(i8ie.tensor(x)).numpy(), tconv(torch.tensor(x)).detach().numpy())

    def test_conv2d2(self):
        iconv = i8ie.Conv2d(10, 20, 3, padding =1)
        tconv = torch.nn.Conv2d(10, 20, 3, padding =1)

        weight = self.get_ndarray((20,10,3,3))
        bias = self.get_ndarray(20)

        tconv.weight.data= torch.tensor(weight)
        tconv.bias.data= torch.tensor(bias)
        iconv.load_weight(weight)
        iconv.load_bias(bias)

        x = self.get_ndarray((30,10, 22, 22))
        self.assertEqualArray(iconv(i8ie.tensor(x)).numpy(), tconv(torch.tensor(x)).detach().numpy())

    def test_conv2d3(self):
        iconv = i8ie.Conv2d(10, 20, 3, stride=7,padding =3)
        tconv = torch.nn.Conv2d(10, 20, 3, stride=7,padding =3)

        weight = self.get_ndarray((20,10,3,3))
        bias = self.get_ndarray(20)

        tconv.weight.data= torch.tensor(weight)
        tconv.bias.data= torch.tensor(bias)
        iconv.load_weight(weight)
        iconv.load_bias(bias)

        x = self.get_ndarray((30,10, 50, 50))
        self.assertEqualArray(iconv(i8ie.tensor(x)).numpy(), tconv(torch.tensor(x)).detach().numpy())

