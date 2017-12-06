import numpy as np
from constants import *
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt

TIME_STEPS = 100
BATCH_SIZE = 100
HIDDEN_SIZE = 30
EPOCH = 200


def load_data():
    X = np.genfromtxt(X_FILE, delimiter=',')
    y = np.genfromtxt(Y_FILE, delimiter=',')
    # X = [[i, math.sin(0.01 * i)] for i in range(2000)]
    # y = [[i, math.cos(0.01 * i)] for i in range(2000)]
    return X, y


def train_test_split(X, y, test_size=0.3):
    train_size = round(len(X) * (1 - test_size))
    minmax = preprocessing.MinMaxScaler((-1, 1))
    X_train = minmax.fit_transform(X[:train_size])
    X_test = minmax.fit_transform(X[train_size:])
    y_train = minmax.fit_transform(y[:train_size])
    y_test = minmax.fit_transform(y[train_size:])
    return X_train, y_train, X_test, y_test


def get_seq_data(X, y):
    seqX = []
    seqY = []
    for i in range(len(X) - TIME_STEPS):
        seqX.append(X[i:i + TIME_STEPS + 1])
        seqY.append(y[i + TIME_STEPS])
    return np.array(seqX), np.array(seqY)


def keras_rnn():
    X, y = load_data()
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train, y_train = get_seq_data(X_train, y_train)
    X_test, y_test = get_seq_data(X_test, y_test)
    IN_SIZE = X_train.shape[-1]
    OUT_SIZE = y_train.shape[-1]
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, input_dim=IN_SIZE))
    model.add(Dense(OUT_SIZE, input_dim=HIDDEN_SIZE))
    model.add(Activation("tanh"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH, validation_split=0.05)
    predicted = model.predict(X_test)
    rmse = mean_squared_error(y_test, predicted)
    r2 = r2_score(y_test, predicted, multioutput='raw_values')
    print(rmse)
    print(r2)
    # plt.scatter([p[0] for p in predicted], [p[1] for p in predicted])
    # plt.scatter([t[0] for t in y_test], [t[1] for t in y_test])
    # plt.scatter(list(range(len(predicted))), predicted)
    # plt.scatter(list(range(len(y_test))), [p[0] for p in y_test])
    # plt.show()


if __name__ == '__main__':
    keras_rnn()
