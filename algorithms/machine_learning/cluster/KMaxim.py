import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances


class KMaximClustering:
    def __init__(self, k=3, radius=0.5, max_iter=100, tol=1e-4):
        self.k = k
        self.radius = radius
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        self.centroids = self._find_centroids(X)
        for _ in range(self.max_iter):
            old_centroids = np.copy(self.centroids)
            self.labels = self._assign_labels(X)
            self.centroids = self._update_centroids(X)
            if np.sum(np.abs(old_centroids - self.centroids)) < self.tol:
                break

    def _find_centroids(self, X):
        distances = euclidean_distances(X)
        density = np.sum(np.exp(-distances ** 2 / (2 * self.radius ** 2)), axis=1)
        sorted_indices = np.argsort(-density)
        centroids = X[sorted_indices[:self.k]]
        return centroids

    def _assign_labels(self, X):
        labels = np.zeros(X.shape[0], dtype=int)
        distances = euclidean_distances(X, self.centroids)
        nearest_centroids = np.argmin(distances, axis=1)
        for i in range(X.shape[0]):
            labels[i] = nearest_centroids[i]
        return labels

    def _update_centroids(self, X):
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            centroids[i] = np.mean(X[self.labels == i], axis=0)
        return centroids
