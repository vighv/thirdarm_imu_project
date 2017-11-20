import numpy as np
import pandas as pd
from constants import *
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
from sklearn.metrics import r2_score

BATCH_START = 0
TIME_STEPS = 30
BATCH_SIZE = 40
HIDDEN_SIZE = 100


def load_data(x_file, y_file, n_steps):
    X = np.genfromtxt(x_file, delimiter=',')
    y = np.genfromtxt(y_file, delimiter=',')
    seqX = []
    seqY = []
    for i in range(len(X)-n_steps):
        seqX.append(X[i:i+n_steps])
        seqY.append(y[i+n_steps])
    return np.array(seqX), np.array(seqY)


def train_test_split(X, y, test_size=0.1):
    train_size = round(len(X) * (1 - test_size))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return X_train, y_train, X_test, y_test


def keras_rnn():
    X, y = load_data(X_FILE, Y_FILE, TIME_STEPS)
    IN_SIZE = X.shape[2]
    OUT_SIZE = y.shape[1]
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, input_dim=IN_SIZE, return_sequences=False))
    model.add(Dense(OUT_SIZE, input_dim=HIDDEN_SIZE))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=8, validation_split=0.1)
    predicted = model.predict(X_test)
    rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
    r2 = r2_score(y_test, predicted)
    print(predicted)
    print(rmse)
    print(r2)


if __name__ == '__main__':
    keras_rnn()
