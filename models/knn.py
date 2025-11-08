# models/knn.py
import numpy as np
from collections import Counter

class KNN:
    """
    k-Nearest Neighbors classifier (from-scratch).
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def _predict_sample(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        X_test_np = np.array(X_test)
        return np.array([self._predict_sample(x) for x in X_test_np])
