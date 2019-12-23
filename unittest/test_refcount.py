import unittest
import i8ie
import numpy as np
import torch
import time

class TestRefcount(unittest.TestCase):
    def get_ndarray(self, shape):
        return np.random.uniform(-100,100,shape).astype(np.float32)

    def test_selfassign(self):
        ndarray = self.get_ndarray((4,4))
        t = i8ie.tensor(ndarray)
        t = t
        t = t
        t = t
        self.assertEqual(t.data.ref_count(), 1)

    def test_release(self):
        ndarray = self.get_ndarray((4,4))
        t = i8ie.tensor(ndarray)
        b = t
        c = t
        c = 0
        self.assertEqual(t.data.ref_count(), 1)
        self.assertEqual(b.data.ref_count(), 1)

    def test_reshape(self):
        ndarray = self.get_ndarray((4,4))
        t = i8ie.tensor(ndarray)
        t.reshape(-1,2)
        c = t.reshape(4,-1)
        self.assertEqual(t.data.ref_count(), 2)
        self.assertEqual(c.data.ref_count(), 2)

    def test_pass_layer(self):
        fc = i8ie.Linear(4, 4)
        ndarray = self.get_ndarray((4,4))
        t = i8ie.tensor(ndarray)
        fc(t)
        t = fc(t)
        u = fc(t)
        t = fc(u)
        self.assertEqual(t.data.ref_count(), 1)
        self.assertEqual(u.data.ref_count(), 1)
