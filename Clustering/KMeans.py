import numpy as np


class KMeans:
    def __init__(self):
        self.means = None
        self.labels = None
        self.x = None

    def fit(self, x, k, early_stop):

        self.x =  x

        # Initialize mean points
        means = x[np.random.randint(0, x.shape[0], size=k)]
        labels = self.clustering(means, x)
        step = 0

        # Start updating
        while step < early_stop:
            new_means = self.update_means(means, labels, x)
            if (new_means == means).all:
                break
            labels = self.clustering(new_means, x)
            means = new_means
            step += 1

        self.means = new_means
        self.labels = labels

    def clustering(self, mean_points, x):

        distance_matrix = []
        for point in mean_points:
            distance = np.sqrt(np.sum((x - point) ** 2, axis=1))
            distance_matrix.append(distance)

        labels = np.argmin(np.array(distance_matrix), axis=0)
        return labels

    def update_means(self, means, labels, x):
        new_means = means
        for label in np.unique(labels):
            new_means[label] = np.mean(x[labels == label], axis=0)
        return new_means

    def knn(self, x, k):
        """
        K Nearest Neighbor
        """

        y = np.zeros(len(x))
        for i in range(len(x)):
            distance = np.sqrt(np.sum((self.x - x[i]) ** 2, axis=1))
            k_nearst_neighbor = self.labels[np.argsort(distance)[:k]]
            y[i] = k_nearst_neighbor[np.argmax(np.bincount(k_nearst_neighbor))]

        return y

