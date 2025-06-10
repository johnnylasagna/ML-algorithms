import numpy as np

class KMeans:

    def __init__(self, data, k):
        self.data = data
        self.k = k

    def initializeClusterCentroids(self):
        mins = np.min(self.data, axis=0)
        maxs = np.max(self.data, axis=0)
        self.centroids = np.array([
            np.random.uniform(mins, maxs)
            for _ in range(self.k)
        ])

    def calculateDistance(self, datapoint, centroid):
        return np.linalg.norm(datapoint - centroid, axis = 1)

    def calculateClosestPoints(self):
        distances = np.linalg.norm(self.data[:, np.newaxis, :] - self.centroids, axis=2)
        self.labels = np.argmin(distances, axis=1)
    
    def newClusters(self):
        new_centroids = np.array([
            self.data[self.labels == i].mean(axis=0) if np.any(self.labels == i) else self.centroids[i]
            for i in range(self.k)
        ])
        self.centroids = new_centroids
    
    def calculateLoss(self):
        assignedCentroids = self.centroids[self.labels]
        distances  = self.calculateDistance(self.data, assignedCentroids)
        loss = np.mean(distances)
        return loss
    
    def train(self, n_iter):
        self.initializeClusterCentroids()
        for i in range(n_iter):
            self.calculateClosestPoints()
            self.calculateLoss()
            self.newClusters()

if __name__=='__main__':
    # Create pseudo data: 2D points in two clusters
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(20, 2))
    cluster2 = np.random.normal(loc=[7, 7], scale=0.5, size=(20, 2))
    data = np.vstack([cluster1, cluster2])

    kmeans = KMeans(data, k=2)
    kmeans.train(n_iter=10)
    labels = kmeans.calculateClosestPoints()
    loss = kmeans.calculateLoss()
    print("Final cluster assignments:", labels)
    print("Final centroids:\n", kmeans.centroids)
    print("Final loss:", loss)
