import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.labels = None
        self.distance = None

    def fit(self, X):
        X = np.array(X)
        n_clusters = len(X)
        # Initialize label & distance
        self.labels = np.arange(n_clusters)
        self.distance = np.zeros([n_clusters, n_clusters])
        for i in range(n_clusters - 1):
            for j in range(i + 1, n_clusters):
                self.distance[i][j] = self.distance[j][i] = np.sqrt(((X[i] - X[j]) ** 2).sum())

        # Start Agglomeration Process
        while n_clusters > self.n_clusters:
            # Find nearest clusters, then merge them
            merger_id, mergee_id = self.find_nearest_cluster()
            # print('===== MERGE {}, {} ======'.format(merger_id, mergee_id))
            self.update_distance(merger_id, mergee_id)
            self.update_label(merger_id, mergee_id)
            n_clusters -= 1

    def find_nearest_cluster(self):
        # Find the minimum nonzero distance of pairs of clusters
        valid_distance = np.ma.MaskedArray(self.distance, self.distance == 0)
        return np.unravel_index(np.argmin(valid_distance), valid_distance.shape)

    def update_distance(self, merger_id, mergee_id):
        """
        When merging two clusters (A, B) to larger C, other clusters or points distance to C will be updated. Here I
        simply choose `single-linkage clustering`, which will simply update by dis(C) = min(dis(A), dis(B)).

        Args:
            merger_id: The point index in X whose cluster will merge the mergee cluster.
            mergee_id: The point index in X whose cluster will be merged into the merger cluster.

        """
        merger_label = self.labels[merger_id]
        mergee_label = self.labels[mergee_id]
        # Pick min value as new distance
        new_distance = self.distance[[merger_id, mergee_id]].min(axis=0)
        # Get all data point's index that is in merger or mergee
        target_id = np.where(np.isin(self.labels, (merger_label, mergee_label)))[0]

        # Update distance by row & column
        self.distance[target_id, :] = new_distance
        self.distance[:, target_id] = new_distance.reshape(-1, 1)

    def update_label(self, merger_id, mergee_id):
        merger_label = self.labels[merger_id]
        mergee_label = self.labels[mergee_id]

        # Change mergee's label to merger's label
        self.labels[self.labels == mergee_label] = merger_label
        # To make the labels in continuous order starting from 0
        self.labels[self.labels > mergee_label] -= 1
