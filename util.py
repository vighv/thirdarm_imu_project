import numpy as np


def test_split(X, y, start, test_size):
    end = start + int(len(X) * test_size)
    X_train = np.vstack((X[:start], X[end:]))
    X_test = X[start:end]
    y_train = np.vstack((y[:start], y[end:]))
    y_test = y[start:end]
    return X_train, X_test, y_train, y_test


def moving_avg_smooth(y_pred, alpha=0.38):
    y_smooth = np.zeros_like(y_pred)
    y_smooth[0] = y_pred[0]
    for i in range(1, y_pred.shape[0]):
        y_smooth[i] = alpha * y_pred[i] + (1 - alpha) * y_smooth[i - 1]
    return y_smooth