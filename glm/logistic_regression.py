import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, with_intercept=True):
        self.with_intercept = with_intercept
        self.theta = None

    def fit(self, X, y, lr=0.1, epochs=100):
        X = np.array(X)
        y = np.expand_dims(np.array(y), axis=1)

        if len(X) != len(y):
            raise ValueError('Invalid input')
        if self.with_intercept:
            X = np.concatenate((np.ones([len(X), 1]), np.array(X)), axis=1)

        self.theta = np.random.rand(X.shape[1], 1)
        loss = self.get_loss(X, y)

        # Gradient Descent
        epoch = 0
        while epoch < epochs:
            loss = self.get_loss(X, y)
            if epoch % 10 == 0:
                print('===== Epoch {} ===== Loss {} ====='.format(epoch, round(loss, 2)))
            self.theta -= lr * self.get_gradient(X, y)
            epoch += 1
        print('===== Epoch {} ===== Loss {} ====='.format(epoch, round(loss, 2)))

    def predict(self, X, threshold=0.5):
        X = np.array(X)
        if self.with_intercept:
            X = np.concatenate((np.ones([len(X), 1]), np.array(X)), axis=1)
        if X.shape[1] != len(self.theta):
            raise ValueError('Invalid input')
        likelihood = self.get_likelihood(X)
        return np.where(likelihood >= threshold, 1, 0)

    def get_likelihood(self, X):
        likelihood = sigmoid(X @ self.theta)
        return likelihood

    def get_loss(self, X, y):
        likelihood = self.get_likelihood(X)
        loss = -(y.T @ np.log(likelihood) + (1 - y).T @ np.log(1 - likelihood)).mean()
        return loss

    def get_gradient(self, X, y):
        gradient = X.T @ (self.get_likelihood(X) - y)
        return gradient

