import tensorflow as tf
import numpy as np
import csv

from matplotlib import pyplot as plt


def load_data():
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

    return X, y


def sigmoid(X, W, b):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    return 1 / (1 + tf.math.exp(-(W @ X + b)))


def softmax(X):
    ym = np.zeros(shape=(150, 3))
    X = tf.transpose(X)

    for i in range(150):
        z1 = X[i][0]
        z2 = X[i][1]
        z3 = X[i][2]

        ym[i][0] = tf.math.exp(z1) / (
            tf.math.exp(z1) + tf.math.exp(z2) + tf.math.exp(z3)
        )
        ym[i][1] = tf.math.exp(z2) / (
            tf.math.exp(z1) + tf.math.exp(z2) + tf.math.exp(z3)
        )
        ym[i][2] = tf.math.exp(z3) / (
            tf.math.exp(z1) + tf.math.exp(z2) + tf.math.exp(z3)
        )

    return tf.convert_to_tensor(tf.transpose(ym), dtype=tf.float64)


def predict(model, X, y):
    W = model["W"]
    b = model["b"]
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]
    l = 1 / (1 + tf.math.exp(-(W @ X + b)))
    h1 = 1 / (1 + tf.math.exp(-(W1 @ l + b1)))
    h2 = 1 / (1 + tf.math.exp(-(W2 @ h1 + b2)))
    ym = tf.nn.softmax(h2, axis=0)
    return ym


def cost_function(parameters, X, y):
    W = parameters[0]
    b = parameters[1]
    W1 = parameters[2]
    b1 = parameters[3]
    W2 = parameters[4]
    b2 = parameters[5]
    l = 1 / (1 + tf.math.exp(-(W @ X + b)))
    h1 = 1 / (1 + tf.math.exp(-(W1 @ l + b1)))
    h2 = 1 / (1 + tf.math.exp(-(W2 @ h1 + b2)))
    ym = tf.nn.softmax(h2, axis=0)
    cost = tf.reduce_sum((y * tf.experimental.numpy.log(ym)), 1, keepdims=True)
    cost = (-1 / 150) * cost
    return ym, cost


def gradient_descent(X, y, num_of_iteration, learning_rate):
    np.random.seed(0)

    W = tf.Variable(
        initial_value=np.random.randn(10, 4),
        dtype=tf.float32,
        trainable=True,
        shape=(10, 4),
    )
    b = tf.Variable(
        initial_value=np.zeros((10, 1)), dtype=tf.float32, trainable=True, shape=(10, 1)
    )
    W1 = tf.Variable(
        initial_value=np.random.randn(100, 10),
        dtype=tf.float32,
        trainable=True,
        shape=(100, 10),
    )
    b1 = tf.Variable(
        initial_value=np.zeros((100, 1)),
        dtype=tf.float32,
        trainable=True,
        shape=(100, 1),
    )
    W2 = tf.Variable(
        initial_value=np.random.randn(3, 100),
        dtype=tf.float32,
        trainable=True,
        shape=(3, 100),
    )
    b2 = tf.Variable(
        initial_value=np.zeros((3, 1)), dtype=tf.float32, trainable=True, shape=(3, 1)
    )

    model = {"W": W, "b": b, "W1": W1, "b1": b1, "W2": W2, "b2": b2}
    parameters = [W, b, W1, b1, W2, b2]
    for i in range(num_of_iteration):
        with tf.GradientTape() as tape:
            tape.watch(parameters)
            ym, cost_func = cost_function(parameters, X, y)

        (
            W_gradient,
            b_gradient,
            W1_gradient,
            b1_gradient,
            W2_gradient,
            b2_gradient,
        ) = tape.gradient(cost_func, parameters)

        W.assign_sub(W_gradient * learning_rate)
        b.assign_sub(b_gradient * learning_rate)
        W1.assign_sub(W1_gradient * learning_rate)
        b1.assign_sub(b1_gradient * learning_rate)
        W2.assign_sub(W2_gradient * learning_rate)
        b2.assign_sub(b2_gradient * learning_rate)

        # Assign new parameters to the model
        model = {"W": W, "b": b, "W1": W1, "b1": b1, "W2": W2, "b2": b2}

        print(cost_function(parameters, X, y))

        del tape

    return ym, model


def compute_training_error(model, X, y):
    classified = 0
    misclassified = 0

    y_pred = np.zeros(shape=(3, 150))
    ym = predict(model, X, y)
    for i in range(150):
        max_index = np.argmax([ym[0][i], ym[1][i], ym[2][i]])
        if max_index == 0:
            y_pred[0][i] = 1
        elif max_index == 1:
            y_pred[1][i] = 1
        else:
            print(f"this is max_index:{max_index}")
            y_pred[2][i] = 1

    for i in range(150):
        if (y[:, i] == y_pred[:, i]).all():
            classified += 1
        else:
            misclassified += 1

    error_ratio = misclassified / 150
    return error_ratio


def visualization(X, y, ym):
    ym = ym.numpy()
    X = X.T
    y = y.T
    ym = ym.T
    svd = np.linalg.svd
    centered_data = X - X.mean(axis=0)
    U, S, Vt = svd(centered_data, full_matrices=False)
    reduced_data = X @ Vt[:2].T
    for i in range(len(y)):
        if y[i].argmax() == ym[i].argmax():
            if y[i].argmax() == 0:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], "ro")
            if y[i].argmax() == 1:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], "go")
            if y[i].argmax() == 2:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], "bo")
        else:
            if y[i].argmax() == 0:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], "ro", mfc="none")
            if y[i].argmax() == 1:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], "go", mfc="none")
            if y[i].argmax() == 2:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], "bo", mfc="none")
    plt.xlabel(" reduced data first coordinate")
    plt.ylabel(" reduced data second coordinate")
    plt.show()


def main():
    X, y = load_data()
    X = X.T
    y = y.T

    ym, model = gradient_descent(X, y, num_of_iteration=500, learning_rate=0.05)
    compute_training_error(model, X, y)

    visualization(X, y, ym)


if __name__ == "__main__":
    main()
