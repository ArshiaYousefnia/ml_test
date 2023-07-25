import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
from MultiLayerPerceptron import MultiLayerPerceptron

data_sheet = pd.read_csv("database_test/classification_data.csv")
x_train = pd.DataFrame.to_numpy(data_sheet.loc[:, 'x1':'x2'], dtype=np.float32)
y_train = pd.DataFrame.to_numpy(data_sheet.loc[:, 'y':'y'], dtype=np.float32)
y_train = y_train.reshape(-1, 1)

inputDim = 2
learningRate = 0.22
epochs = 11000

model = MultiLayerPerceptron(inputDim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=0.0001)

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
        current = np.array([j * 10 / heat_map_resolution, i * 10 / heat_map_resolution], dtype=np.float32)
        heat_map_data[i, j] = model(torch.from_numpy(current)).data.numpy()[0] * 100


zero_values = []
one_values = []
for i in range(len(y_train)):
    if y_train[i] == 0:
        zero_values.append(x_train[i])
    else:
        one_values.append(x_train[i])

x_train0 = np.array(zero_values, dtype=np.float32)
x_train0 = x_train0.reshape(-1, 2)
x_train1 = np.array(one_values, dtype=np.float32)
x_train1 = x_train1.reshape(-1, 2)

plt.clf()
plt.plot(x_train0[:, 0] * heat_map_resolution / 10, x_train0[:, 1] * heat_map_resolution / 10, '.')
plt.plot(x_train1[:, 0] * heat_map_resolution / 10, x_train1[:, 1] * heat_map_resolution / 10, '.')
plt.ylim(0, heat_map_resolution)
plt.imshow(heat_map_data, cmap='BrBG_r')
plt.colorbar()
plt.show()

plt.clf()
plt.plot([i for i in range(epochs)], losses)
plt.show()
