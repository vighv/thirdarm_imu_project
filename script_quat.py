from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn import neighbors
from sklearn import preprocessing
import numpy as np
from sklearn import linear_model

XFILE = 'data/imu_proc09-Nov-2017.csv'
YFILE = 'data/vicon_proc09-Nov-2017.csv'

def swap_cols(arr, frm, to):
    arr[:,[frm,to]] = arr[:,[to,frm]]

def data_proc(X_train, y_train, choice='minmax'):
    # Choice whether minmax or standard
    if choice =='minmax':
        minmax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        X = minmax.fit_transform(X_train)
        y = minmax.fit_transform(y_train)
    elif choice == 'standard':
        X = preprocessing.scale(X_train)
        y = preprocessing.scale(y_train)

    return X,y

def svr():
    print('SVM Regression: Linear')
    minmax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_train = np.genfromtxt(XFILE, delimiter=',')

    # IMU data swap columns: First 10 is upper arm, next 10 is torso
    # for i in range(10):
    #     swap_cols(X_train,i,10+i)

    X = minmax.fit_transform(X_train)

    y_train = np.genfromtxt(YFILE, delimiter=',')
    y = minmax.fit_transform(y_train)

    clf = SVR('linear')
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))

    clf = SVR('rbf')
    print('RBF Kernel')
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))


def mlp():
    print('Multilayer Perceptron')
    minmax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_train = np.genfromtxt(XFILE, delimiter=',')

    # IMU data swap columns: First 10 is upper arm, next 10 is torso
    # for i in range(10):
    #     swap_cols(X_train,i,10+i)

    X = minmax.fit_transform(X_train)

    y_train = np.genfromtxt(YFILE, delimiter=',')
    y = minmax.fit_transform(y_train)

    clf = MLPRegressor(max_iter=500, solver='adam', activation='relu')
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))

def knn():
    print('KNN Regression')
    wts = 'distance'
    nbrs = 10

    X_train = np.genfromtxt(XFILE, delimiter=',')
    y_train = np.genfromtxt(YFILE, delimiter=',')

    X,y = data_proc(X_train, y_train, 'standard')

    clf = neighbors.KNeighborsRegressor(nbrs, weights=wts, algorithm='auto')
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))


def ridge():
    print('Ridge Regression')

    X_train = np.genfromtxt(XFILE, delimiter=',')
    y_train = np.genfromtxt(YFILE, delimiter=',')

    X,y = data_proc(X_train, y_train)

    clf = linear_model.Ridge (alpha = 0.01)

    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))


def TSCV(splits=3):
    #Time Series Cross Validation
    # First try with RBF SVR

    X_train = np.genfromtxt(XFILE, delimiter=',')
    y_train = np.genfromtxt(YFILE, delimiter=',')

    X,y = data_proc(X_train, y_train, 'minmax')

    tscv = TimeSeriesSplit(n_splits=splits)

    svr_rbf = SVR('rbf')

    for i in range(y.shape[1]):
        for train_index, test_index in tscv.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            svr_rbf.fit(X_train, y_train[:,i])
            y_pred = svr_rbf.predict(X_test)

            score = svr_rbf.score(X_test, y_test[:,i])
            print score


if __name__ == '__main__':
    #mlp()
    #svr()
    #knn()
    #ridge()
    TSCV(5)
