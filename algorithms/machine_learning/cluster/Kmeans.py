
from sklearn.cluster import KMeans


class KMeansClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, data):
        self.kmeans.fit(data)
        self.labels = self.kmeans.labels_
        self.centers = self.kmeans.cluster_centers_

    def predict(self, data):
        return self.kmeans.predict(data)

