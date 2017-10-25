from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA

XFILE = 'data/imu_proc17-Oct-2017.csv'
YFILE = 'data/vicon_proc17-Oct-2017.csv'


def mlp():
    X = preprocessing.scale(np.genfromtxt(XFILE, delimiter=','))
    y = preprocessing.scale(np.genfromtxt(YFILE, delimiter=','))
    clf = MLPRegressor(max_iter=500, hidden_layer_sizes=10, alpha=10)
    print('=== MLP ===')
    kfold(clf, X, y)


def svr_pca():
    print('=== linear SVR ===')
    X = preprocessing.scale(np.genfromtxt(XFILE, delimiter=','))
    y = preprocessing.scale(np.genfromtxt(YFILE, delimiter=','))
    for n in range(1, X.shape[1]):
        print(str(n) + ' dimension')
        pca = PCA(n)
        X_reduced = pca.fit_transform(X)
        clf = SVR('linear')
        kfold(clf, X_reduced, y)

    print('=== RBF SVR ===')
    for n in range(1, X.shape[1]):
        print(str(n) + ' dimension')
        pca = PCA(n)
        X_reduced = pca.fit_transform(X)
        clf = SVR()
        kfold(clf, X_reduced, y)


def mlp_pca():
    X = preprocessing.scale(np.genfromtxt(XFILE, delimiter=','))
    y = preprocessing.scale(np.genfromtxt(YFILE, delimiter=','))

    print('=== MLP ===')
    for n in range(1, X.shape[1]):
        # print(str(n) + ' dimension')
        pca = PCA(n)
        X_reduced = pca.fit_transform(X)
        clf = MLPRegressor(max_iter=500, alpha=10)
        kfold(clf, X_reduced, y)


def kfold(clf, X, y):
    mean = []
    std = []
    for i in range(y.shape[1]):
        scores = cross_val_score(clf, X, y[:, i], cv=10, n_jobs=-1)
        mean.append(np.mean(scores))
        std.append(np.std(scores))
    print('\t'.join(map(str, mean)))
    # print('\t'.join(map(str, std)))


if __name__ == '__main__':
    mlp()
    # svr_pca()
