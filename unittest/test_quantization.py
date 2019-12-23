import unittest
import i8ie
import numpy as np
import torch

class TestQuantization(unittest.TestCase):
    def get_ndarray(self, shape):
        return np.random.uniform(-1,1,shape).astype(np.float32)

    def assertEqualArray(self, a, b):
        return self.assertTrue(np.allclose(a,b, atol=0.1))

    def test_quantize(self):
        ndarray = self.get_ndarray((4,4))
        t = i8ie.tensor(ndarray)
        q = i8ie.quantize(t, 0.025, 100)
        self.assertEqualArray(ndarray, (q.numpy().astype(np.float32)-100) * 0.025)

    def test_dequantize(self):
        ndarray = self.get_ndarray((4,4))
        t = i8ie.tensor(ndarray)
        q = i8ie.quantize(t, 0.025, 100)
        self.assertEqualArray(ndarray, i8ie.dequantize(q).numpy())
