import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# task 1.0
# ------- 2d data -----------
raw_data2d = np.load("data2d.npz")
X = raw_data2d["X"]
# print(X)
y = raw_data2d["y"]
for i in range(len(y)):
    if y[i] == 0:  # class 0 with green
        plt.plot(X[i, 0], X[i, 1], "ro")
    else:
        plt.plot(X[i, 0], X[i, 1], "go")

plt.xlabel("2d data first coordinate")
plt.ylabel("2d data second coordinate")
plt.show()

# ------- 5d data -----------
raw_data5d = np.load("data5d.npz")
X = raw_data5d["X"]
# print(X)
y = raw_data5d["y"]

for i in range(len(y)):
    if y[i] == 0:  # class 0 with green
        plt.plot(X[i, 0], X[i, 1], "ro")
    else:
        plt.plot(X[i, 0], X[i, 1], "go")

plt.xlabel("5d data first coordinate")
plt.ylabel("5d data second coordinate")
plt.show()
