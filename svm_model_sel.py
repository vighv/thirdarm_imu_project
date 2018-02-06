from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split

XFILE = 'data/imu_proc01-Feb-2018.csv'
YFILE = 'data/vicon_proc01-Feb-2018.csv'

if __name__ == '__main__':
    X = np.genfromtxt(XFILE, delimiter=',')
    y = np.genfromtxt(YFILE, delimiter=',')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1, 'auto']
    param_grid = {'C': Cs, 'gamma': gammas}
    for yd in range(y_train.shape[1]):
        print('Searching params for y_' + str(yd) + '...')
        grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, 'r2', n_jobs=-1, cv=10)
        grid_search.fit(X_train, y_train[:, yd])
        best_params = grid_search.best_params_
        print(best_params)
