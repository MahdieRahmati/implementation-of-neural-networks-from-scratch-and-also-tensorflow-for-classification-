import numpy as np
from matplotlib import pyplot as plt


def Phi(X, W, b):
    return 1 / (1 + np.exp(-(X @ W + b)))


def C(W, b, X, y):
    cost = 0
    for i in range(len(X)):
        Xi = np.reshape(X[i], (1, len(X[i])))
        if y[i] == 0:
            cost += -np.log(1 - (1 / (1 + np.exp(-(W * Xi.T + b)))))
        else:  # y[i] = 1
            cost += -np.log(1 / (1 + np.exp(-(W * Xi.T + b))))
    print("cost: {cost}".format(cost=cost))
    return cost


def compute_dC_dw(W, b, X, y):
    gradient = np.zeros((1, X.shape[1]), dtype=np.float32)  # 1*m
    for i in range(len(X)):
        Xi = np.reshape(X[i], (1, len(X[i])))
        if y[i] == 0:
            g = Xi.T * np.exp(W * Xi.T + b) / (np.exp(W * Xi.T + b) + 1)
            gradient += g.T

        else:  # y[i] = 1
            g = -Xi.T /  (np.exp(W * Xi.T + b) + 1)
            gradient += g.T

    return gradient.T  # m*n


def compute_dC_db(W, b, X, y):
    gradient = 0
    for i in range(len(X)):
        Xi = np.reshape(X[i], (1, len(X[i])))
        if y[i] == 0:
            gradient += np.exp(Xi @ W + b) / ((np.exp(Xi @ W + b) + 1))
        else:  # yi = 1
            gradient += -1 / ((np.exp(Xi @ W + b) + 1))
    return gradient[0]


def gradient_descent(start_W, start_b, learn_rate, X, y, n_iter=100, tolerance=1e-06):
    W = start_W  # w = m*1
    b = start_b
    for _ in range(n_iter):
        C(W, b, X, y)
        visualization(W, b, X, y)

        diff1 = -learn_rate * compute_dC_dw(W, b, X, y)
        W += diff1

        diff2 = -learn_rate * compute_dC_db(W, b, X, y)
        b += diff2

        if np.all(np.abs(diff1) <= tolerance) and np.all(np.abs(diff2) <= tolerance):
            break

    return W, b


def c(W, b, Xi, yi):
    X = np.reshape(Xi, (1, len(Xi)))
    if yi == 0:
        return -np.log(1 - (1 / (1 + np.exp(-(W * X.T + b)))))
    else:  # y[i] = 1
        return -np.log(1 / (1 + np.exp(-(W * X.T + b))))


def compute_dC_dw_for_one_X(W, b, Xi, yi):

    X = np.reshape(Xi, (1, len(Xi)))
    if yi == 0:
        g = X.T * np.exp(W * X.T + b) / ((np.exp(W * X.T + b) + 1))
        return g.T
    else:  # yi = 1
        g = -X.T / ((np.exp(W * X.T + b) + 1))
        return g.T


def compute_dC_dw_for_one_b(W, b, Xi, yi):
    gradient = 0
    X = np.reshape(Xi, (1, len(Xi)))
    if yi == 0:
        gradient += np.exp(X @ W + b) / ((np.exp(X @ W + b) + 1))
    else:  # yi = 1
        gradient += -1 / ((np.exp(X @ W + b) + 1))
    return gradient[0]


def compute_dC_dw_numeric(W, b, X, y):
    print()
    print()
    #  choose a random x from data
    m = len(X[0])
    n = len(X)
    index = np.random.randint(0, n - 1)
    Xi = X[index]
    yi = y[index]

    gt = compute_dC_dw_for_one_X(W, b, Xi, yi)
    print("gt")
    print(gt)
    print()

    dlt = 1e-7
    g = np.zeros(m)
    for i in range(m):
        Xi2 = Xi.copy()
        Xi2[i] += dlt
        print(c(W, b, Xi2, yi))
        print(c(W, b, Xi, yi))
        print()
        g = (c(W, b, Xi2, yi) - c(W, b, Xi, yi)) / dlt
        print(g)
        print(np.linalg.norm(gt - g))  # must be small


def compute_dC_db_numeric(W, b, X, y):
    print()
    print()
    #  choose a random x from data
    m = len(X[0])
    n = len(X)
    index = np.random.randint(0, n - 1)
    Xi = X[index]
    yi = y[index]

    gt = compute_dC_dw_for_one_b(W, b, Xi, yi)
    print("gt")
    print(gt)
    print()

    dlt = 1e-7
    g = np.zeros(m)
    for i in range(m):
        Xi2 = Xi.copy()
        Xi2[i] += dlt
        print(c(W, b, Xi2, yi))
        print(c(W, b, Xi, yi))
        print()
        g = (c(W, b, Xi2, yi) - c(W, b, Xi, yi)) / (2 * dlt)
        print(g)
        print(np.linalg.norm(gt - g))  # most be small


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


def visualization(W, b, X, y):

    for i in range(len(X)):
        phi = Phi(np.reshape(X[i], (1, len(X[i]))), W, b)
        if phi < 0.5:
            y2 = 0
            if y[i] == y2:
                plt.plot(X[i][0], X[i][1], "o", color="blue", markersize=3)
            else:
                plt.plot(X[i][0], X[i][1], "o", mfc="none", color="blue", markersize=3)
        else:
            y2 = 1
            if y[i] == y2:
                plt.plot(X[i][0], X[i][1], "o", color="red", markersize=3)
            else:
                plt.plot(X[i][0], X[i][1], "o", mfc="none", color="red", markersize=3)

    Xs = X[:, 0]
    Y = (-Xs * W[0] - b) / W[1]
    plt.plot(Xs, Y, color="green")
    plt.xlim(-4, 4)
    plt.ylim(-6, 6)
    plt.draw()
    plt.pause(0.1)
    plt.cla()


def main():
    raw_data = np.load("data2d.npz")
    X = raw_data["X"]
    y = raw_data["y"]

    start_W = np.abs(np.random.randn(X.shape[1], 1))
    start_b = np.abs(np.random.randn(1))
    learning_rate = 0.001
    W, b = gradient_descent(
        start_W=start_W, start_b=start_b, learn_rate=learning_rate, X=X, y=y, n_iter=100
    )

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("W: {w}".format(w=W))
    print("b: {b}".format(b=b))
    print("training error: {error}".format(error=compute_training_error(X, y, W, b)))
    visualization(W, b, X, y)


if __name__ == "__main__":
    main()
