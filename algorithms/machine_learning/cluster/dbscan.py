from sklearn.cluster import DBSCAN
class DBSCANClustering:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def fit(self, data):
        self.dbscan.fit(data)
        self.labels = self.dbscan.labels_

    def predict(self, data):
        return self.dbscan.fit_predict(data)