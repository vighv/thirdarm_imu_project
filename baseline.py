from sklearn.base import BaseEstimator, RegressorMixin


class MeanRegressor(BaseEstimator, RegressorMixin):
    """An example of classifier"""

    def __init__(self):
        """
        Called when initializing the classifier
        """
        self.average = 0

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """
        self.average = y.mean(0)

    def predict(self, X):
        return self.average

    def score(self, X, y=None):
        return self.score(X, y)
