from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

XFILE = 'data/imu_proc01-Feb-2018.csv'
YFILE = 'data/vicon_proc01-Feb-2018.csv'

X = np.genfromtxt(XFILE, delimiter=',')
y = np.genfromtxt(YFILE, delimiter=',')


def pairwise_plot(pred, X_test, y_test):
    for xd in range(X_test.shape[1]):
        plt.scatter(X_test[:, xd], y_test[:, yd], c='r')
        plt.scatter(X_test[:, xd], pred, c='b')
        plt.show()


if __name__ == '__main__':
    mlp = MLPRegressor(20, max_iter=500, alpha=10)
    svr = SVR()
    svr_lin = SVR(kernel='linear')
    svr_poly = SVR(kernel='poly')
    bayesian = BayesianRidge()
    models = [svr, svr_lin, svr_poly, bayesian, mlp]
    names = ['SVM-RBF', 'SVM-linear', 'SVM-poly', 'Bayesian', 'MLP']
    for m, clf in enumerate(models):
        # cross-validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        r2_valid = []
        rmse_valid = []

        for yd in range(y_train.shape[1]):
            y_1d = y_train[:, yd]
            score = np.mean(cross_val_score(clf, X_train, y_1d, scoring='r2', cv=10, n_jobs=-1))
            r2_valid.append(score)
            score = np.mean(cross_val_score(clf, X_train, y_1d, scoring='neg_mean_squared_error', cv=10, n_jobs=-1))
            rmse_valid.append(score)

        r2_test = []
        rmse_test = []
        for yd in range(y_test.shape[1]):
            y_1d = y_test[:, yd]
            clf.fit(X_train, y_train[:, yd])
            pred = clf.predict(X_test)
            r2 = r2_score(y_1d, pred)
            mse = mean_squared_error(y_1d, pred)
            r2_test.append(r2)
            rmse_test.append(mse)

        print('===' + names[m] + '===')
        print('Validation:')
        print('\t'.join(map(str, r2_valid)))
        print('\t'.join(map(str, rmse_valid)))
        print('Test:')
        print('\t'.join(map(str, r2_test)))
        print('\t'.join(map(str, rmse_test)))
