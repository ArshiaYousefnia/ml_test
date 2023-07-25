import pandas as pd
import numpy as np

data = pd.read_csv("classification_data.csv")
columns = pd.DataFrame.to_numpy(data.loc[:, 'x1':'x2'], dtype=np.float32)

x_values = [[1, 1], [2, 8], [3, 5]]

x_train = np.array(x_values, dtype=np.float32)
#x_train = x_train.reshape(-1, 2)

print(columns)
print(x_train)

