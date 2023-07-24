import torch
from network import NaiveNet

data = [[1, 2], [9, 12]]
x_data = torch.tensor(data)

print(x_data)

print(x_data[1][0])

one_data = torch.ones_like(x_data)

print(one_data + one_data)
print(one_data)

print(NaiveNet())
