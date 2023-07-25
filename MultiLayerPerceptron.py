import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size):
        super(MultiLayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(input_size, 5)
        self.linear2 = nn.Linear(5, 4)
        self.linear3 = nn.Linear(4, 1)

        self.linear1.zero_grad()
        self.linear2.zero_grad()
        self.linear3.zero_grad()

    def forward(self, x):
        output1 = torch.sigmoid(self.linear1(x))
        output2 = torch.sigmoid(self.linear2(output1))
        output3 = torch.sigmoid(self.linear3(output2))
        return output3

test_model = MultiLayerPerceptron(2)

for i in test_model.parameters():
    print(i)