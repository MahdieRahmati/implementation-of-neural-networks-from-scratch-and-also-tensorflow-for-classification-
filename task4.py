import numpy as np
import csv
import matplotlib.pyplot as plt

X = np.zeros(shape=(150, 4))
y = np.zeros(shape=(150, 3))
with open("iris.data") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count != 150:
            X[line_count][0] = row[0]
            X[line_count][1] = row[1]
            X[line_count][2] = row[2]
            X[line_count][3] = row[3]
            if row[4] == "Iris-setosa":
                y[line_count] = [1, 0, 0]
            elif row[4] == "Iris-versicolor":
                y[line_count] = [0, 1, 0]
            else:
                y[line_count] = [0, 0, 1]
            line_count += 1

svd = np.linalg.svd
centered_data = X - X.mean(axis=0)
U, S, Vt = svd(centered_data, full_matrices=False)
reduced_data = X @ Vt[:2].T

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "ro")
plt.xlabel("2d data first coordinate")
plt.ylabel("2d data second coordinate")
plt.show()
