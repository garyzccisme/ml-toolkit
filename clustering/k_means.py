import numpy as np


class KMeans:
    def __init__(self):
        self.means = None
        self.labels = None
        self.X = None

    def fit(self, X, k, early_stop):
        self.X = X

        # Initial means points are randomly selected from range(number of data points)
        means = X[np.random.choice(X.shape[0], size=k, replace=False)]
        labels = self.classify(means)
        step = 0

        # Start updating
        while step < early_stop:
            new_means = self.update_means(labels)
            # Convergent
            if (new_means == means).all:
                break
            labels = self.classify(new_means)
            means = new_means
            step += 1

        self.means = means
        self.labels = labels

    def classify(self, means):
        """
        Do classification by given mean points. The data point shares the same label with the nearest mean point.

        Args:
            means: Array in shape [k, X.shape[1]].

        Returns: Array in shape [X.shape[0]], values in [0, k - 1].

        """
        distance_matrix = []
        for point in means:
            # Use Euclidean distance
            distance = np.sqrt(np.sum((self.X - point) ** 2, axis=1))
            distance_matrix.append(distance)

        labels = np.argmin(np.array(distance_matrix), axis=0)
        return labels

    def update_means(self, labels):
        """
        Update mean points by given labels. The new mean is the mean of points with the same label.

        Args:
            labels: Array in shape [X.shape[0]], values in [0, k - 1].

        Returns: Array in shape [k, X.shape[1]].

        """
        new_means = []
        for label in range(max(labels) + 1):
            new_means.append(np.mean(self.X[labels == label], axis=0))
        return np.array(new_means)

    def knn(self, x, k):
        """
        K Nearest Neighbor Method
        """

        y = np.zeros(len(x))
        for i in range(len(x)):
            distance = np.sqrt(np.sum((self.X - x[i]) ** 2, axis=1))
            k_nearest_neighbor = self.labels[np.argsort(distance)[:k]]
            y[i] = np.bincount(k_nearest_neighbor).argmax()
        return y

