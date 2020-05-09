import numpy as np


class LinearRegression:
    def __init__(self, with_intercept=True):
        self.with_intercept = with_intercept
        self.beta = None

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError('Invalid input')
        if self.with_intercept:
            X = np.stack((np.ones(len(X)), np.array(X)))
        self.beta = (X.T @ X).I @ X.T @ y

    def predict(self, X):
        X = np.array(X)
        if self.with_intercept:
            X = np.stack((np.ones(len(X)), np.array(X)))
        if X.shape[1] != len(self.beta):
            raise ValueError('Invalid input')
        y_pred = X @ self.beta
        return y_pred

