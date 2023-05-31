import numpy as np
import tensorflow as tf
import task1


def Phi(X, W, b):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    return 1 / (1 + tf.math.exp(-(X @ W + b)))


def C(W, b, X, y):
    c = 0
    for i in range(len(X)):
        Xi = np.reshape(X[i], (1, len(X[i])))
        if y[i] == 0:
            c += -tf.experimental.numpy.log(
                1 - (1 / (1 + tf.math.exp(-(W * Xi.T + b))))
            )
        else:  # y[i] = 1
            c += -tf.experimental.numpy.log(1 / (1 + tf.math.exp(-(W * Xi.T + b))))
    print(c)
    return c


def gradient_descent_using_tf(
    start_W, start_b, learning_rate, X, y, n_iter=50, tolerance=1e-06
):
    W = tf.Variable(
        initial_value=start_W, dtype=tf.float32, trainable=True, shape=(len(X[0]), 1)
    )
    b = tf.Variable(initial_value=start_b, dtype=tf.float32, trainable=True)

    for _ in range(n_iter):

        C(W, b, X, y)
        task1.visualization(W, b, X, y)
        with tf.GradientTape(persistent=True) as tape:
            # Find prediction value and calculate loss value

            cost_function = C(W, b, X, y)

        # Calculate partial derivative by each parameter
        W_gradient = tape.gradient(cost_function, W)
        b_gradient = tape.gradient(cost_function, b)

        # update value of each parameter: w1 = w0 - learning_rate * d(loss)/dw
        W.assign_sub(W_gradient * learning_rate)
        b.assign_sub(b_gradient * learning_rate)

        print("W in iteration:")
        print(W)
        print("b in iteration:")
        print(b)

        del tape

    return W, b


def compute_training_error(X, y, W, b):
    classified = 0
    misclassified = 0

    for i in range(len(X)):
        phi = Phi(np.reshape(X[i], (1, len(X[i]))), W, b)
        if phi < 0.5:
            y2 = 0
        else:
            y2 = 1

        if y[i] == y2:
            classified += 1
        else:
            misclassified += 1

    error_ratio = misclassified / len(X)
    return error_ratio


def main():
    raw_data = np.load("data2d.npz")
    X = raw_data["X"]
    y = raw_data["y"]

    start_W = np.random.randn(X.shape[1], 1)
    start_b = np.random.randn(1)
    learning_rate = 0.01
    W, b = gradient_descent_using_tf(
        start_W=start_W,
        start_b=start_b,
        learning_rate=learning_rate,
        X=X,
        y=y,
        n_iter=20,
    )

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("W: {w}".format(w=W))
    print("b: {b}".format(b=b))
    print("training error: {error}".format(error=compute_training_error(X, y, W, b)))
    print(compute_training_error(X, y, W, b))


if __name__ == "__main__":
    main()
