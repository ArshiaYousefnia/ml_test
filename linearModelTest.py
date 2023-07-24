#test
import torch
import numpy as np
from torch.autograd import Variable
from linearRegression import LinearRegression
import matplotlib.pyplot as plt

x_values = [[1, 2], [1.5, 2.5], [2, 3], [5, 6], [8, 9], [5, 5], [6, 9], [1, 1], [1, 4], [1, 8], [1.5, 6], [4, 2], [3, 4], [6, 1], [8, 2], [4, 6], [4, 8], [4, 9], [3, 9], [6.5, 4], [6, 6], [8, 6], [9, 1], [9, 4], [5, 7], [7, 7]]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 2)

y_values = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

inputDim = 2
outputDim = 1
learningRate = 0.01
epochs = 100

model = LinearRegression(inputDim, outputDim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

losses = []

for epoch in range(epochs):
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, labels)
    print(loss)
    loss.backward()
    optimizer.step()

    print("epoch {}, loss {}".format(epoch, loss.item()))
    losses.append(loss.item())

test = [1, 2]
test_data = np.array(test, dtype=np.float32)

with torch.no_grad():
    border = model(Variable(torch.from_numpy(test_data))).data.numpy()
    print(border)

heat_map_resolution = 200
heat_map_data = np.zeros((heat_map_resolution, heat_map_resolution))
for i in range(heat_map_resolution):
    for j in range(heat_map_resolution):
        current = np.array([i * 10 / heat_map_resolution, j * 10 / heat_map_resolution], dtype=np.float32)
        heat_map_data[i, j] = model(torch.from_numpy(current)).data.numpy()[0]

plt.clf()
plt.plot(x_train[:, 0] * heat_map_resolution / 10, x_train[:, 1] * heat_map_resolution / 10, '.')#, label='data', alpha=0.5)
plt.ylim(0, heat_map_resolution)
plt.imshow(heat_map_data, cmap='BuGn')
plt.colorbar()
plt.show()

plt.clf()
plt.plot([i for i in range(epochs)], losses)
plt.show()