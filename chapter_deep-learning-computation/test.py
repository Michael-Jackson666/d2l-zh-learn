import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

X = torch.rand(2, 784)
print(net(X))