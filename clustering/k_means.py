import numpy as np


class KMeans:
    def __init__(self):
        self.centroids = None
        self.labels = None
        self.X = None

    def fit(self, X, k, early_stop):
        self.X = X

        # Initial centroids points are randomly selected from range(number of data points)
        centroids = X[np.random.choice(X.shape[0], size=k, replace=False)]
        labels = self.classify(centroids)
        step = 0

        # Start updating
        while step < early_stop:
            new_centroids = self.update_centroids(labels)
            # Convergent
            if (new_centroids == centroids).all:
                break
            labels = self.classify(new_centroids)
            centroids = new_centroids
            step += 1

        self.centroids = centroids
        self.labels = labels

    def classify(self, centroids):
        """
        Do classification by given centroids points. The data point shares the same label with the nearest centroids point.

        Args:
            centroids: Array in shape [k, X.shape[1]].

        Returns: Array in shape [X.shape[0]], values in [0, k - 1].

        """
        distance_matrix = []
        for point in centroids:
            # Use Euclidean distance
            distance = np.sqrt(np.sum((self.X - point) ** 2, axis=1))
            distance_matrix.append(distance)

        labels = np.argmin(np.array(distance_matrix), axis=0)
        return labels

    def update_centroids(self, labels):
        """
        Update centroids points by given labels. The new centroid is the mean of points with the same label.

        Args:
            labels: Array in shape [X.shape[0]], values in [0, k - 1].

        Returns: Array in shape [k, X.shape[1]].

        """
        new_centroids = []
        for label in range(max(labels) + 1):
            new_centroids.append(np.mean(self.X[labels == label], axis=0))
        return np.array(new_centroids)

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

