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
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        x = x.reshape(-1, 784)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

#torch.set_num_threads(1)

model = Net()
model.load_state_dict(torch.load('../model28.pt'))

batch_size = 100
transform = transforms.Compose(
    [transforms.ToTensor()])
test_dataset = torchvision.datasets.MNIST(root='../data/mnist/', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

xs = []
ts = []
for batch_idx, (x, target) in enumerate(test_loader):
    xs.append(x)
    ts.append(target)

correct_cnt = 0
tt = 0
for x,target in zip(xs,ts):
    t = time.time()
    x = model(x).data.numpy()
    x[0] = x[0]
    tt += time.time() - t
    p = np.argmax(x, axis = 1)
    correct_cnt += (p  == target.numpy()).sum()
print(tt,'sec')
print(correct_cnt)


w = model.state_dict()['fc.weight'].numpy()
b = model.state_dict()['fc.bias'].numpy()
fc = tensor_core.Linear(w,b)
correct_cnt = 0
tt = 0
for x,target in zip(xs,ts):
    x = x.reshape(-1,784).numpy()
    x = tensor_core.create(x)
    t = time.time()
    x = fc(x).numpy()
    tt += time.time() - t
    p = np.argmax(x, axis = 1)
    correct_cnt += (p == target.numpy()).sum()

print(correct_cnt)
print(tt,'sec')
