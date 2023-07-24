import torch
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.linear.zero_grad()

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        #out = self.linear(x)
        return out
