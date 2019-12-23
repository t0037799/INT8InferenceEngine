import unittest
import i8ie
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MyNet(i8ie.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = i8ie.Conv2d(1, 20, kernel_size=5)
        self.conv2 = i8ie.Conv2d(20, 50, kernel_size=5)
        self.fc1 = i8ie.Linear(800, 500)
        self.fc2 = i8ie.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = i8ie.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = i8ie.max_pool2d(x, kernel_size=2, stride=2)
        x = x.reshape(-1, 800)
        x = i8ie.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestQuantizedLayer(unittest.TestCase):
    def get_ndarray(self, shape):
        return np.random.uniform(-2,2,size=shape).astype(np.float32)
    def setUp(self):
        torch.set_num_threads(1)
        state_dict = torch.load('conv28.pt')
        self.torch_model = Net()
        self.torch_model.load_state_dict(state_dict)
        self.my_model = MyNet()
        self.my_model.load(state_dict)
        self.my_model.prepare()
        self.my_model(i8ie.tensor(self.get_ndarray((100,1,28,28))))
        self.my_model.convert()

    def assertEqualArray(self, a, b):
        flag = (np.isclose(a,b, rtol = 0.3).sum() > 0.8 * a.size)
        return self.assertTrue(flag)

    def test_each_layer(self):
        x = self.get_ndarray((10,1,28,28))
        q = self.my_model.conv1(i8ie.quantize(i8ie.tensor(x),0.025,127))
        x = self.torch_model.conv1(torch.tensor(x))
        d = i8ie.dequantize(q)
        self.assertEqualArray(x.detach().numpy(), d.numpy())

        q = i8ie.max_pool2d(q, kernel_size=2, stride=2)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        d = i8ie.dequantize(q)
        self.assertEqualArray(x.detach().numpy(), d.numpy())

        q = self.my_model.conv2(q)
        x = self.torch_model.conv2(x)
        d = i8ie.dequantize(q)
        self.assertEqualArray(x.detach().numpy(), d.numpy())

        q = i8ie.max_pool2d(q, kernel_size=2, stride=2)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        d = i8ie.dequantize(q)
        self.assertEqualArray(x.detach().numpy(), d.numpy())

        q = q.reshape(-1, 800)
        x = x.reshape(-1, 800)
        q = i8ie.relu(self.my_model.fc1(q))
        x = F.relu(self.torch_model.fc1(x))
        d = i8ie.dequantize(q)
        self.assertEqualArray(x.detach().numpy(), d.numpy())

        q = self.my_model.fc2(q)
        x = self.torch_model.fc2(x)
        d = i8ie.dequantize(q)
        self.assertEqualArray(x.detach().numpy(), d.numpy())
