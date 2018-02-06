from sklearn import preprocessing
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import math

XFILE = 'data/imu_proc29-Jan-2018.csv'
YFILE = 'data/vicon_proc29-Jan-2018.csv'
# XFILE = 'data/imu_proc09-Nov-2017.csv'
# YFILE = 'data/vicon_proc09-Nov-2017.csv'

TRAIN_SIZE = 0.9
CROSS_VAL = 10

X = np.genfromtxt(XFILE, delimiter=',')
y = np.genfromtxt(YFILE, delimiter=',')


def pairwise_plot(pred, X_test, y_test):
    for xd in range(X_test.shape[1]):
        plt.scatter(X_test[:, xd], y_test[:, yd], c='r')
        plt.scatter(X_test[:, xd], pred, c='b')
        plt.show()


mlp = MLPRegressor(20, max_iter=500, alpha=10)
svr = SVR()
bayesian = BayesianRidge()
models = [bayesian, mlp, svr]
names = ['Bayesian', 'MLP', 'SVM']
for m, clf in enumerate(models):
    # cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    r2 = []
    rmse = []

    for yd in range(y_train.shape[1]):
        clf.fit(X_train, y_train[:, yd])
        pred = clf.predict(X_test)
        r2.append(r2_score(y_test[:, yd], pred))
        rmse.append(mean_squared_error(y_test[:, yd], pred))

    print('===' + names[m] + '===')
    print('\t'.join(map(str, r2)))
    print('\t'.join(map(str, rmse)))
    plt.show()
