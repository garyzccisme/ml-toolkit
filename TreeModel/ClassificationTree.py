import numpy as np


class Leaf:
    def __init__(self, label: int):
        self.label = label


class Node:
    def __init__(self, threshold, column_idx, left, right):
        self.threshold = threshold
        self.column_idx = column_idx
        self.left = left
        self.right = right


class ClassificationTree:

    def __int__(self):
        self.tree = None

    def fit(self, x, y):
        self.tree = self.grow(x, y)

    def grow(self, x, y):

        best_gini = self.gini(y)
        if best_gini == 0:
            return Leaf(label=y[0])

        for column_idx in range(x.shape[1]):
            for potential_threshold in np.unique(x[:, column_idx]):

                # Note: np.where returns tuple of arrays
                left_index = np.where(x[:, column_idx] <= potential_threshold)[0]
                right_index = np.where(x[:, column_idx] > potential_threshold)[0]

                left_gini = self.gini(y[left_index])
                right_gini = self.gini(y[right_index])
                gini = len(left_index) / len(y) * left_gini + len(right_index) / len(y) * right_gini

                if gini < best_gini:
                    best_gini = gini
                    best_column_idx, best_threshold = column_idx, potential_threshold
                    left_x, left_y = x[left_index, :], y[left_index]
                    right_x, right_y = x[right_index, :], y[right_index]

        root = Node(threshold=best_threshold, column_idx=best_column_idx,
                    left=self.grow(left_x, left_y), right=self.grow(right_x, right_y))

        return root

    def gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        return 1 - ((counts / len(y)) ** 2).sum()

    def predict(self, x, node=None):

        if node is None:
            node = self.tree

        if isinstance(node, Leaf):
            return np.ones(len(x)) * node.label

        left_index = np.where(x[:, node.column_idx] <= node.threshold)
        right_index = np.where(x[:, node.column_idx] > node.threshold)

        y = np.zeros(len(x))
        y[left_index] = self.predict(x[left_index], node.left)
        y[right_index] = self.predict(x[right_index], node.right)
        return y






