import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import time
import numpy as np
import tensor_core


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
        return F.log_softmax(x, dim=1)

class MyNet():
    def __init__(self):
        self.is_quant = False
        self.conv1 = tensor_core.Conv2d(1, 20, 5)
        self.conv2 = tensor_core.Conv2d(20, 50, 5)
        self.fc1 = tensor_core.Linear(800, 500)
        self.fc2 = tensor_core.Linear(500, 10)

    def load(self, state_dict):
        for key in state_dict:
            name, attr = key.split('.')
            if attr == 'weight':
                self.__dict__[name].load_weight(state_dict[key])
            elif attr == 'bias':
                self.__dict__[name].load_bias(state_dict[key])

    def __call__(self, x):
        x = tensor_core.create(x)
        if self.is_quant:
            x = tensor_core.quantize(x, 0.01, 50)
        x = self.conv1(x)
        t = time.time()
        x = tensor_core.maxpool2d(x, 2, 2)
        x = self.conv2(x)
        x = tensor_core.maxpool2d(x, 2, 2)
        x = x.reshape([-1,800])
        x = self.fc1(x)
        x = tensor_core.relu(x)
        x = self.fc2(x)
        if self.is_quant:
            x = tensor_core.dequantize(x)
        x = x.numpy()
        return x

    def prepare(self):
        for attr in self.__dict__:
            if attr != 'is_quant':
                self.__dict__[attr].prepare()

    def convert(self):
        for attr in self.__dict__:
            if attr != 'is_quant':
                self.__dict__[attr].convert()
        self.is_quant = True


#torch.set_num_threads(1)
state_dict = torch.load('conv28.pt')
torch_model = Net()
torch_model.load_state_dict(state_dict)
my_model = MyNet()
my_model.load(state_dict)

batch_size = 100

transform = transforms.Compose(
    [transforms.ToTensor()])
test_dataset = torchvision.datasets.MNIST(root='./data/mnist/', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

xs = []
ts = []
for batch_idx, (x, target) in enumerate(test_loader):
    xs.append(x)
    ts.append(target)

t = time.time()
correct_cnt = 0
for x,target in zip(xs,ts):
    x = my_model(x)
    p = np.argmax(x, axis = 1)
    correct_cnt += (p == target.numpy()).sum()

print('my float:',time.time() - t,'sec')
print(correct_cnt)

t = time.time()
correct_cnt = 0
my_model.prepare()
for x,target in zip(xs,ts):
    x = my_model(x)
    p = np.argmax(x, axis = 1)
    correct_cnt += (p == target.numpy()).sum()
    break
my_model.convert()
print('calibrat:', time.time() - t,'sec')
print(correct_cnt)


t = time.time()
correct_cnt = 0
for x,target in zip(xs,ts):
    x = my_model(x)
    p = np.argmax(x, axis = 1)
    correct_cnt += (p == target.numpy()).sum()

print('my int:',time.time() - t,'sec')
print(correct_cnt)


t = time.time()
correct_cnt = 0
for x,target in zip(xs,ts):
    out = torch_model(x)
    _, pred_label = torch.max(out.data, 1)
    correct_cnt += (pred_label == target.data).sum()
print('torch:',time.time() - t,'sec')
print(correct_cnt)

