from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from pca import PCA
from kmeans import KMeans
from visualization import Visualization
from gaussian import Gaussian

class Assignment1:
    def __init__(self):
        self.pca = PCA()
        self.kmeans = KMeans()
        self.visualization = Visualization()
        self.gaussian = Gaussian()

    def digits(self):
        # Load the digits dataset
        digits = load_digits()
        X = digits.data
        y = digits.target

        # perform principal component analysis (PCA) on the dataset
        pca_X = self.pca.pca(X, y)

        # plot the original data and the PCA projection
        self.visualization.pca_plotting(X, pca_X, y)

        # perform K-means clustering on the projected data and visualize the clusters
        k_values = [2,3,4,5,6,7,8,9,10]
        silhouette_scores = []

        
        for k in k_values:
            pred, clusters = self.kmeans.kmeans(pca_X, k)
            score = silhouette_score(pca_X, pred)
            silhouette_scores.append(score)
            self.visualization.plot_kmeans_clusters(pca_X, pred, clusters, k)

        # plot silhouette score for K-means
        self.visualization.plot_silhouette_scores(k_values, silhouette_scores, "K-Means Clustering Silhouette Score")

        
        # clear silhouette scores for GMM
        silhouette_scores = []

        # run GMM and plot
        for k in k_values:
            pred, means, covariances, weights = self.gaussian.gaussian(pca_X, k)
            self.visualization.plot_gmm_clusters(pca_X, pred, means, k)
            score = self.silhouette(pca_X, pred)
            silhouette_scores.append(score)

        # plot silhouette score for GMM
        self.visualization.plot_silhouette_scores(k_values, silhouette_scores, "Gaussian Mixture Model Clustering Silhouette Score")
        

    def silhouette(self, X, pred):
        score = silhouette_score(X, pred)
        return score

    def blob(self):
        # load the blob dataset
        X, y = make_blobs(n_samples=1000, centers=4, random_state=42, cluster_std=1.0)
        print(X.shape, y.shape)

        # perform K-means clustering on the projected data and visualize the clusters
        k_values = [2,3,4,5,6,7,8,9,10]
        silhouette_scores = []

        for k in k_values:
            pred, clusters = self.kmeans.kmeans(X, k)
            score = silhouette_score(X, pred)
            silhouette_scores.append(score)
            self.visualization.plot_kmeans_clusters(X, pred, clusters, k)

        # plot silhouette score for K-means
        self.visualization.plot_silhouette_scores(k_values, silhouette_scores, "K-Means Clustering Silhouette Score")

        # run GMM and plot
        silhouette_scores = []
        for k in k_values:
            pred, means, covariances, weights = self.gaussian.gaussian(X, k)
            self.visualization.plot_gmm_clusters(X, pred, means, k)
            score = self.silhouette(X, pred)
            silhouette_scores.append(score)

        # plot silhouette score for GMM
        self.visualization.plot_silhouette_scores(k_values, silhouette_scores, "Gaussian Mixture Model Clustering Silhouette Score")


    def optimized_pca_analysis(self, X, y):
        # perform optimized PCA on the dataset
        pca_X = self.pca.optimized_pca(X, y)


if __name__ == "__main__":
    assignment = Assignment1()
    assignment.digits()
    assignment.blob()
    assignment.optimized_pca_analysis(load_digits().data, load_digits().target)