from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import math

XFILE = 'data/imu_proc01-Feb-2018.csv'
YFILE = 'data/vicon_proc01-Feb-2018.csv'

X = np.genfromtxt(XFILE, delimiter=',')
y = np.genfromtxt(YFILE, delimiter=',')

if __name__ == '__main__':
    y = np.append(y, np.arange(y.shape[0])[..., None], 1)  # keep track of time
    mlp = MLPRegressor(20, max_iter=500, alpha=10)
    svr_opt = SVR(gamma=0.1, C=100)
    bayesian = BayesianRidge(alpha_1=1e-04, alpha_2=1e-07, lambda_1=1e-08, lambda_2=1e-04)
    models = [svr_opt, bayesian, mlp]
    names = ['SVM-RBF', 'Bayesian Ridge', 'MLP']

    for i, clf in enumerate(models):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # Sort test data in time series
        y_test_idx = y_test[:, -1]
        sorted_idx = np.argsort(y_test_idx)
        y_test = np.array([y_test[i] for i in sorted_idx])
        X_test = np.array([X_test[i] for i in sorted_idx])
        y_test_idx.sort()

        # Delete time indices in the last column
        y_train = np.delete(y_train, -1, 1)
        y_test = np.delete(y_test, -1, 1)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Obtain test accuracy
        r2_test = []
        rmse_indiv = []
        for yd in range(y_test.shape[1]):
            print('Fitting y_' + str(yd) + ' for ' + names[i] + '...')
            y_1d = y_test[:, yd]
            clf.fit(X_train, y_train[:, yd])
            pred = clf.predict(X_test)
            r2 = r2_score(y_1d, pred)
            label = 'y_' + str(yd + 1)

            # Plot actual values
            # label_true = label + '-true'
            # label_pred = label + '-predicted'
            # plt.plot(y_test_idx, y_1d, label=label_true)
            # plt.plot(y_test_idx, pred, alpha=0.7, label=label_pred)
            # plt.title(names[i])
            # plt.xlabel('Time index')
            # plt.ylabel('Value')
            # plt.legend()
            # plt.show()

            # Plot RMSE
            r2_test.append(r2)
            rmse_list = np.sqrt((y_1d - pred)**2)
            rmse_indiv.append(rmse_list)
            plt.plot(y_test_idx, rmse_list, label=label, alpha=0.7)
        print('===' + names[i] + '===')
        print('Test:')
        print('\t'.join(map(str, r2_test)))

        # Plot RMSE
        plt.title(names[i])
        plt.xlabel('Time index')
        plt.ylabel('Root Mean Squared Error')
        plt.legend()
        plt.show()
