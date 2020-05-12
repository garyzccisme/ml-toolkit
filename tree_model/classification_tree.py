import numpy as np


class Leaf:
    def __init__(self, label: int):
        self.label = label


class Node:
    def __init__(self, threshold=None, column_idx=None, left=None, right=None):
        self.threshold = threshold
        self.column_idx = column_idx
        self.left = left
        self.right = right


class ClassificationTree:

    def __init__(self):
        self.tree = Node()

    @staticmethod
    def gini(y):
        _, counts = np.unique(y, return_counts=True)
        return 1 - ((counts / len(y)) ** 2).sum()

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree = self.grow(X, y)

    def grow(self, X, y):

        # Initialization
        best_gini = self.gini(y)
        best_threshold = None
        best_column_idx = None
        left_x, left_y = None, None
        right_x, right_y = None, None

        if best_gini == 0:
            return Leaf(label=y[0])

        # Search all possible split points in X
        for column_idx in range(X.shape[1]):
            for potential_threshold in np.unique(X[:, column_idx]):

                # Note: np.where returns tuple of arrays
                left_index = np.nonzero(X[:, column_idx] <= potential_threshold)[0]
                right_index = np.nonzero(X[:, column_idx] > potential_threshold)[0]

                left_gini = self.gini(y[left_index])
                right_gini = self.gini(y[right_index])
                gini = (len(left_index) * left_gini + len(right_index) * right_gini) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_column_idx, best_threshold = column_idx, potential_threshold
                    left_x, left_y = X[left_index, :], y[left_index]
                    right_x, right_y = X[right_index, :], y[right_index]

        root = Node(threshold=best_threshold, column_idx=best_column_idx,
                    left=self.grow(left_x, left_y), right=self.grow(right_x, right_y))

        return root

    def predict(self, X, node=None):

        X = np.array(X)
        if node is None:
            node = self.tree
        if isinstance(node, Leaf):
            return np.ones(len(X)) * node.label

        left_index = np.where(X[:, node.column_idx] <= node.threshold)
        right_index = np.where(X[:, node.column_idx] > node.threshold)

        y = np.zeros(len(X))
        y[left_index] = self.predict(X[left_index], node.left)
        y[right_index] = self.predict(X[right_index], node.right)
        return y






