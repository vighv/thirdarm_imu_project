from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

XFILE = 'data/imu_proc01-Feb-2018.csv'
YFILE = 'data/vicon_proc01-Feb-2018.csv'

X = np.genfromtxt(XFILE, delimiter=',')
X = X[:, 10:]
y = np.genfromtxt(YFILE, delimiter=',')


def val_line_plot(x, y_true, y_pred, yd):
    # Plot actual values and prediction in line plot
    label = 'y_' + str(yd + 1)
    label_true = label + ' true'
    label_pred = label + ' predicted'

    plt.plot(x, y_true, label=label_true)
    plt.plot(x, y_pred, alpha=0.7, label=label_pred)
    plt.title(names[i])
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def true_line_pred_scatter(x, y_test, y_pred, yd, title):
    # Plot all test data as line plot and prediction as scatter
    label = 'y_' + str(yd + 1)
    label_true = label + ' true'
    label_pred = label + ' predicted'
    plt.plot(np.arange(y_test.shape[0]) / 30, y_test[:, yd], label=label_true)
    plt.scatter(x, y_pred, 8, 'r', alpha=0.7, label=label_pred)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def true_vs_pred_scatter(y_true, y_pred, yd, title):
    # Plot real value against prediction
    label = 'y_' + str(yd + 1)
    label_true = label + ' true'
    label_pred = label + ' predicted'
    plt.scatter(pred, y_true, 8)
    plt.xlabel(label_pred)
    plt.ylabel(label_true)
    plt.title(title)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_label = 'R2 = ' + str(round(r2, 3))
    rmse_label = 'RMSE = ' + str(round(rmse, 3))
    xmin = min(y_pred)
    xmax = max(y_pred)
    ymin = min(y_true)
    ymax = max(y_true)
    xrange = xmax - xmin
    yrange = ymax - ymin
    plt.text(xrange * 0.8 + xmin, yrange * 0.5 + ymin, r2_label)
    plt.text(xrange * 0.8 + xmin, yrange * 0.4 + ymin, rmse_label)
    plt.savefig('results/' + title + '_y' + str(yd) + '_truevspred_wristonly.png')
    plt.show()


if __name__ == '__main__':
    y = np.append(y, np.arange(y.shape[0])[..., None], 1)  # keep track of time
    mlp = MLPRegressor(200, max_iter=1000, alpha=0.001)
    svr_opt = SVR(gamma=0.1, C=100)
    # bayesian = BayesianRidge(alpha_1=1e-04, alpha_2=1e-07, lambda_1=1e-08, lambda_2=1e-04)
    models = [svr_opt, mlp]
    names = ['SVM', 'MLP']

    for i, clf in enumerate(models):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # Sort test data in time series
        y_test_idx = y_test[:, -1]
        sorted_idx = y_test_idx.argsort()
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
            y_true = y_test[:, yd]
            clf.fit(X_train, y_train[:, yd])
            pred = clf.predict(X_test)
            r2 = r2_score(y_true, pred)
            r2_test.append(r2)
            # true_vs_pred_scatter(y_true, pred, yd, names[i])

            # Plot RMSE
            label = 'y_' + str(yd + 1)
            r2_test.append(r2)
            rmse_list = np.sqrt((y_true - pred)**2)
            rmse_indiv.append(rmse_list)
            plt.plot(y_test_idx / 30, rmse_list, label=label, alpha=0.7)
            mean_label = label + ' mean'
            y_mean = np.full(y_test_idx.shape, rmse_list.mean())
            print(rmse_list.mean(), rmse_list.std())
            plt.plot(y_test_idx / 30, y_mean, label=mean_label, linestyle='--')

        print('===' + names[i] + '===')
        print('Test:')
        print('\t'.join(map(str, r2_test)))

        # Plot RMSE
        plt.title(names[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Root Mean Squared Error')
        plt.legend()
        plt.show()
