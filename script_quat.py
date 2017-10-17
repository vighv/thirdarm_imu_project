from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import numpy as np

XFILE = 'data/imu_proc17-Oct-2017.csv'
YFILE = 'data/vicon_proc17-Oct-2017.csv'

def swap_cols(arr, frm, to):
    arr[:,[frm,to]] = arr[:,[to,frm]]

def svr():
    minmax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_train = np.genfromtxt(XFILE, delimiter=',')

    # IMU data swap columns: First 10 is upper arm, next 10 is torso
    for i in range(10):
        swap_cols(X_train,i,10+i)

    X = minmax.fit_transform(X_train)

    y_train = np.genfromtxt(YFILE, delimiter=',')
    y = minmax.fit_transform(y_train)

    clf = SVR('linear')
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))
    clf = SVR()

    print 'RBF SVM'
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))


def mlp():
    minmax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_train = np.genfromtxt(XFILE, delimiter=',')

    # IMU data swap columns: First 10 is upper arm, next 10 is torso
    for i in range(10):
        swap_cols(X_train,i,10+i)

    X = minmax.fit_transform(X_train)

    y_train = np.genfromtxt(YFILE, delimiter=',')
    y = minmax.fit_transform(y_train)

    clf = MLPRegressor(max_iter=500)
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        print(np.mean(scores), np.std(scores))

if __name__ == '__main__':
    #mlp()
    svr()
