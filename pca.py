import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self):
        pass
    
    def pca(self, X, y):
        # Load the digits dataset
        print(X.shape, y.shape)
    
    
        # 1. calculate the mean
        total_sum = 0.0
        total_count = X.shape[0] * X.shape[1]
    
        for sample in X:
            for pixel_value in sample:
                total_sum += pixel_value
    
        mean_X = total_sum / total_count
    
        print("Mean pixel value:", mean_X)
    
        # 2. subtract the mean from each value
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] -= mean_X
    
        # 3. calculate the covariance matrix
        cov_matrix = np.cov(X.T)
        print("Covariance matrix:\n", cov_matrix)
    
        # 4. calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        print("Eigenvalues:\n", eigenvalues)
        print("Eigenvectors:\n", eigenvectors)
    
        # 5. choose components and form a feature vector
        idx = np.argsort(eigenvalues) [::-1]
        eigenvectors = eigenvectors[:, idx]

        # 6. choose the top 2 eigenvectors
        feature_vector = eigenvectors[:, :2]  
    
        # 7. project original data into the new feature space
        X_projected = X.dot(feature_vector)
        print("Projected data shape:", X_projected.shape)
        return X_projected

    def optimized_pca(self, X, y):
        # Load the digits dataset
        print(X.shape, y.shape)
    
        # 1. calculate the mean
        mean_X = np.mean(X)
        print("Mean pixel value:", mean_X)
    
        # 2. subtract the mean from each value
        X_centered = X - mean_X
    
        # 3. calculate the covariance matrix
        cov_matrix = np.cov(X_centered.T)
        print("Covariance matrix:\n", cov_matrix)
    
        # 4. calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        print("Eigenvalues:\n", eigenvalues)
        print("Eigenvectors:\n", eigenvectors)

        # 5. sort eigenvectors and eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    
        # 6. calculate the proportion of variance for each principal component
        explained_variance = eigenvalues / np.sum(eigenvalues)

        cumulative_variance = np.cumsum(explained_variance)

        # 7. Find the number of principal components that explain 90% of the variance
        n_components = np.argmax(cumulative_variance >= 0.9) + 1

        print(f"Number of components explaining 90% variance: {n_components}")

        # 8. Project the data onto the top n_components eigenvectors
        X_projected = X_centered @ eigenvectors[:, :n_components]
        print("Projected data shape:", X_projected.shape)

        # plot the cumulative explained variance

        plt.plot(
            range(1, len(eigenvalues) + 1),
            cumulative_variance,
            marker='o'
        )

        plt.axhline(y=0.90, linestyle='--')

        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Proportion of Variance")
        plt.title("PCA Cumulative Explained Variance")

        plt.show()

        return X_projected