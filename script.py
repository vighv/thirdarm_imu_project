from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import numpy as np

XFILE = 'data/imu_proc10-Oct-2017.csv'
YFILE = 'data/vicon_proc10-Oct-2017.csv'


def svr():
    X = preprocessing.scale(np.genfromtxt(XFILE, delimiter=','))
    y = preprocessing.scale(np.genfromtxt(YFILE, delimiter=','))
    clf = SVR('linear')
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))
    clf = SVR()
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))


def mlp():
    X = preprocessing.scale(np.genfromtxt(XFILE, delimiter=','))
    y = preprocessing.scale(np.genfromtxt(YFILE, delimiter=','))
    clf = MLPRegressor(max_iter=500)
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))


if __name__ == '__main__':
    mlp()
    #svr()
