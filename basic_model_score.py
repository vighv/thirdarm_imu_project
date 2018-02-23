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
X = X[:, :10]  # arm only
# X = X[:, 10:]  # wrist only
# X = np.hstack((X[:, :4], X[:, 10:14]))  # angle only
# X = np.hstack((X[:, 4:7], X[:, 14:17]))  # velocity only
# X = np.hstack((X[:, 7:10], X[:, 17:]))  # acceleration only
y = np.genfromtxt(YFILE, delimiter=',')


def pairwise_plot(pred, X_test, y_test):
    for xd in range(X_test.shape[1]):
        plt.scatter(X_test[:, xd], y_test[:, yd], c='r')
        plt.scatter(X_test[:, xd], pred, c='b')
        plt.show()


if __name__ == '__main__':
    mlp = MLPRegressor(200, 'tanh', max_iter=1400, alpha=0.001)
    svr_opt = SVR(gamma=0.1, C=100)
    bayesian = BayesianRidge(alpha_1=1e-04, alpha_2=1e-07, lambda_1=1e-08, lambda_2=1e-04)
    models = [svr_opt, bayesian, mlp]
    names = ['SVM-RBF', 'Bayesian', 'MLP']
    for m, clf in enumerate(models):
        # cross-validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        r2_valid = []
        rmse_valid = []

        # for yd in range(y_train.shape[1]):
        #     y_1d = y_train[:, yd]
        #     score = np.mean(cross_val_score(clf, X_train, y_1d, scoring='r2', cv=10, n_jobs=-1))
        #     r2_valid.append(score)
        #     score = np.mean(cross_val_score(clf, X_train, y_1d, scoring='neg_mean_squared_error', cv=10, n_jobs=-1))
        #     rmse_valid.append(score)

        # Obtain test accuracy
        r2_test = []
        rmse_test = []
        rmse_indiv = []
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
