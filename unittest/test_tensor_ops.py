import unittest
import i8ie
import numpy as np
import torch

class TestTensorOps(unittest.TestCase):
    def get_ndarray(self, shape):
        return np.random.uniform(-100,100,shape).astype(np.float32)

    def assertEqualArray(self, a, b):
        return self.assertTrue(np.array_equal(a,b))

    def test_from_numpy(self):
        ndarray = self.get_ndarray()
        tensor = i8ie.tensor(ndarray)
        self.assertEqualArray(tensor.numpy(), ndarray)

    def test_reshape(self):
        ndarray = self.get_ndarray()
        tensor = i8ie.tensor(ndarray)
        self.assertEqualArray(tensor.numpy(), ndarray)
        self.assertEqualArray(tensor.reshape(-1,2), tensor.reshape(8,-1))
        self.assertEqualArray(tensor.reshape(8,2), tensor.reshape(8,-1))
        self.assertEqualArray(tensor.reshape(-2,4).numpy(), ndarray.reshape(4,-2))
        self.assertEqualArray(tensor.reshape(-2,4).shape, ndarray.reshape(4,-2).shape)

    def test_sum(self):
        ndarray = self.get_ndarray()
        tensor = i8ie.tensor(ndarray)
        self.assertEqualArray(tensor.sum(), ndarray.sum())

    def test_argmax(self):
        ndarray = self.get_ndarray().reshape(-1,4)
        tensor = i8ie.tensor(ndarray)
        self.assertEqualArray(i8ie.argmax(tensor,0).numpy(), np.argmax(ndarray,0))
        self.assertEqualArray(i8ie.argmax(tensor,1).numpy(), np.argmax(ndarray,1))
        self.assertEqualArray(i8ie.argmax(tensor,0), i8ie.tensor(np.array([3.,7,11,15])))
        self.assertEqualArray(i8ie.argmax(tensor,1), i8ie.tensor(np.array([12.,13,14,15])))

    def test_max_pool2d(self):
        ndarray = self.get_ndarray().reshape(1,1,-1,4)
        itensor = i8ie.tensor(ndarray)
        ttensor = torch.tensor(ndarray)
        self.assertEqualArray(i8ie.max_pool2d(itensor, 2,2).numpy(), torch.nn.functional.max_pool2d(ttensor, 2, 2))
        self.assertEqualArray(i8ie.max_pool2d(itensor, 2,1).numpy(), torch.nn.functional.max_pool2d(ttensor, 2, 1))
        self.assertEqualArray(i8ie.max_pool2d(itensor, 1,2).numpy(), torch.nn.functional.max_pool2d(ttensor, 1, 2))
