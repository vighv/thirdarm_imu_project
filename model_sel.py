from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor

XFILE = 'data/imu_proc01-Feb-2018.csv'
YFILE = 'data/vicon_proc01-Feb-2018.csv'


def svm_model_sel(X_train, y_train):
    # Cs = [0.01, 0.1, 1, 10, 20]
    # kernels = ['rbf', 'linear', 'poly']
    Cs = [10, 50, 100, 500, 1000]
    # gammas = [0.001, 0.01, 0.1, 1, 'auto']
    epsilons = [0.0001, 0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'epsilon': epsilons}
    model = SVR(gamma=0.1)
    param_search(X_train, y_train, model, param_grid)


def bayesian_model_sel(X_train, y_train):
    alpha1s = [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10]
    alpha2s = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    lambda1s = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    lambda2s = [1e-6, 1e-5, 1e-4, 0.001, 0.01]
    param_grid = {'alpha_1': alpha1s, 'alpha_2': alpha2s,
                  'lambda_1': lambda1s, 'lambda_2': lambda2s}
    model = BayesianRidge()
    param_search(X_train, y_train, model, param_grid)


def mlp_model_sel(X_train, y_train):
    hidden_sizes = [30, 100, 300, 500]
    # activations = ['relu', 'tanh', 'logistic']
    alphas = [0.0001, 0.001, 0.01, 0.1, 1]
    max_iters = [1000, 2000]
    param_grid = {'max_iter': max_iters, 'hidden_layer_sizes': hidden_sizes,
                  'alpha': alphas}
    model = MLPRegressor(activation='tanh')
    param_search(X_train, y_train, model, param_grid)


def param_search(X_train, y_train, model, param_grid):
    for yd in range(y_train.shape[1]):
        print('Searching params for y_' + str(yd) + '...')
        grid_search = GridSearchCV(model, param_grid, 'r2', n_jobs=-1, cv=10)
        grid_search.fit(X_train, y_train[:, yd])
        best_params = grid_search.best_params_
        print(best_params)


if __name__ == '__main__':
    X = np.genfromtxt(XFILE, delimiter=',')
    y = np.genfromtxt(YFILE, delimiter=',')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # print('===SVM===')
    # svm_model_sel(X_train, y_train)
    print('===Bayesian Ridge===')
    bayesian_model_sel(X_train, y_train)
    # print('===MLP===')
    # mlp_model_sel(X_train, y_train)
