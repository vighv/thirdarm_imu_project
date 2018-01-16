from sklearn import preprocessing
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

XFILE = 'data/imu_proc17-Oct-2017.csv'
YFILE = 'data/vicon_proc17-Oct-2017.csv'
TRAIN_SIZE = 0.9
CROSS_VAL = 10


X = np.genfromtxt(XFILE, delimiter=',')
y = np.genfromtxt(YFILE, delimiter=',')

r2_list = []
mlp = MLPRegressor(20, max_iter=500, alpha=10)
svr = SVR()
bayesian = BayesianRidge()
models = [bayesian, mlp, svr]
names = ['Bayesian', 'MLP', 'SVM']
for m, clf in enumerate(models):
    for _ in range(CROSS_VAL):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
        r2 = []
        for d in range(y_train.shape[1]):
            clf.fit(X_train, y_train[:, d])
            pred = clf.predict(X_test)
            r2.append(r2_score(y_test[:, d], pred))
        r2_list.append(r2)
    print('===' + names[m] + '===')
    print('\t'.join(map(str, np.average(r2_list, 0))))
