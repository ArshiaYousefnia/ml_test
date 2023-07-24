import torch
from torch import nn


class NaiveNet(nn.Module):
    def __init__(self):
        super(NaiveNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        output = self.forward(x)

        return torch.argmax(output, 1)

    def train(self, X, y):
        pass

    
