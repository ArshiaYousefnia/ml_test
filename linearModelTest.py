#test
import torch
import numpy as np
from torch.autograd import Variable
from linearRegression import LinearRegression
import matplotlib.pyplot as plt

x_values = [[1, 2], [1.5, 2.5], [2, 3], [5, 6], [8, 9], [5, 5], [6, 9], [1, 1]]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 2)

y_values = [1, 1, 1, 0, 0, 0, 0, 1]
#y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

inputDim = 2
outputDim = 1
learningRate = 0.01
epochs = 2

model = LinearRegression(inputDim, outputDim)
criterion = torch.nn.CrossEntropyLoss()
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


#plt.clf()
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.plot(range(epochs), losses, label='loss', alpha=0.9)
#plt.legend(loc='best')
#plt.show()
heat_map_data = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        current = np.array([i / 10, j / 10], dtype=np.float32)
        heat_map_data[i, j] = model(torch.from_numpy(current)).data.numpy()[0]

plt.clf()
plt.plot(x_train[:, 0] * 10, x_train[:, 1] * 10, '.', label='data', alpha=0.5)
#plt.plot(test, label='test', alpha=0.5)
plt.imshow(heat_map_data, cmap='autumn')
plt.show()

plt.clf()
plt.plot([i for i in range(epochs)], losses)
plt.show()
