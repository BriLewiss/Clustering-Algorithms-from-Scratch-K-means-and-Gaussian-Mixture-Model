import numpy as np
from sklearn.metrics import silhouette_score

class KMeans:

    def __init__(self, max_iters=100, tol=1e-4):
        self.max_iters = max_iters
        self.tol = tol

    def kmeans(self, X, k):
        # perform K-means clustering on the projected data
        # initialize random centroids
        clusters = {}
        np.random.seed(23)

        for idx in range(k):
            center = 2 * (2*np.random.rand(X.shape[1]) - 1)
            points = []
            cluster = {
                'center' : center,
                'points' : points
            }
            clusters[idx] = cluster

        print("Initial clusters:", clusters)

        '''
        assigning, updating, and predicting the cluster centers
        old_centers: stores the current centroids before updating them
        assign_clusters: assigns each data point to its nearest centroid
        update_clusters: recalculates each centroid as the mean of the points assigned to its cluster
        np.allclose: checks if the old and new centroids are close enough to stop the iteration
        pred_cluster: predicts the final cluster label for each data point using the converged centroids
        
        source: https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/
        '''

        for _ in range(self.max_iters):
            old_centers = [clusters[i]['center'].copy() for i in range(k)]

            clusters = self.assign_clusters(X, clusters, k)
            clusters = self.update_clusters(X, clusters, k)

            new_centers = [clusters[i]['center'] for i in range(k)]

            # check for convergence
            if np.allclose(old_centers, new_centers, atol=self.tol):
                break

        pred = self.pred_cluster(X, clusters, k)
        return pred, clusters


    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # assign points to the nearest centroid
    def assign_clusters(self, X, clusters, k):

        # clear previous assignments
        for i in range(k):
            clusters[i]['points'] = []

        for idx in range(X.shape[0]):
            dist = []

            curr_x = X[idx]

            for i in range(k):
                dis = self.euclidean_distance(curr_x, clusters[i]['center'])
                dist.append(dis)
            curr_cluster = np.argmin(dist)
            clusters[curr_cluster]['points'].append(curr_x)
        return clusters

    # update the centroids based on the average of the points assigned to each cluster
    def update_clusters(self, X, clusters, k):
        for i in range(k):
            points = np.array(clusters[i]['points'])
            if points.shape[0] > 0:
                new_center = points.mean(axis = 0)
                clusters[i]['center'] = new_center

                clusters[i]['points'] = []
        return clusters

    # predict the cluster for each data point based on the final centroid
    def pred_cluster(self, X, clusters, k):
        pred = []
        for i in range(X.shape[0]):
            dist = []
            for j in range(k):
                dist.append(self.euclidean_distance(X[i], clusters[j]['center']))
            pred.append(np.argmin(dist))
        return pred
