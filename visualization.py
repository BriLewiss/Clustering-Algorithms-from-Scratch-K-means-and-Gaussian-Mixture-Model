import matplotlib.pyplot as plt
import numpy as np

class Visualization: 
    def plot_kmeans_clusters(self, X, pred, clusters, k):
        # plot data points with predicted cluster centers
        plt.scatter(X[:,0], X[:,1], c=pred)
        for i in clusters:
            center = clusters[i]['center']
            plt.scatter(center[0], center[1], marker = '^', c = 'red')
            plt.title(f'K-Means Clustering with k={k}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
        plt.show()

    def plot_silhouette_scores(self, k_values, silhouette_scores, title):
        # plot silhouette score
        plt.plot(k_values, silhouette_scores, marker = 'o', color="g")
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title(title)
        plt.show()

    def plot_gmm_clusters(self, X, pred, means, k):
        plt.scatter(X[:,0], X[:,1], c=pred)
        plt.scatter(means[:,0], means[:,1], marker = '^', c = 'red')
        plt.title(f'Gaussian Mixture Model Clustering with k={k}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    def pca_plotting(self, X, pca_X, y):
        # plot the PCA projection of the data
        plt.scatter(pca_X[:,0], pca_X[:,1], c=y)
        plt.title('PCA Projection of the Digits Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

        # show the digits images
        fig, axes = plt.subplots(1,5)
        for i, ax in enumerate(axes):
            ax.imshow(X[i].reshape(8,8), cmap='gray')
            ax.set_title(f'Label: {y[i]}')
            ax.axis('off')

        plt.show()

    def pca_optimized_plotting(self, X, pca_X, y):
        # plot the PCA projection of the data
        plt.scatter(pca_X[:,0], pca_X[:,1], c=y)
        plt.title('Optimized PCA Projection of the Digits Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

        # show the digits images
        fig, axes = plt.subplots(1,5)
        for i, ax in enumerate(axes):
            ax.imshow(X[i].reshape(8,8), cmap='gray')
            ax.set_title(f'Label: {y[i]}')
            ax.axis('off')

        plt.show()

