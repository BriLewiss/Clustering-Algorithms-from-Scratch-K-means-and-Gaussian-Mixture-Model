import numpy as np
from scipy.stats import multivariate_normal

class Gaussian:
    def __init__(self):
        self.max_iter = 100
        self.tol = 1e-4

    def gaussian(self, X, k):
        '''
        Our goal is to find the parameters of the GMM (means, covariances, and mixing coefficients)
        that will best explain the observed data.
        '''
        np.random.seed(0)

        # initialize the parameters
        means, covariances, weights = self.init_params(X, k)

        for i in range(self.max_iter):
            old_means = means.copy()
            # E-step
            responsibilities = self.e_step(X, means, covariances, weights, k)
            # M-step
            means, covariances, weights = self.m_step(X, responsibilities, k)

            # check for convergence
            if np.allclose(old_means, means, atol=self.tol):
                break

        # assign each point to the Gaussian
        pred = np.argmax(responsibilities, axis=1)

        return pred, means, covariances, weights



    def init_params(self, X, k):
        # initialize the parameters of means (from random points in the dataset), covariance matrices, and mixing weights/coefficients
        np.random.seed(0)

        # randomly choose k data points as initial means
        random_indices = np.random.choice(X.shape[0], k, replace=False)
        means = X[random_indices].copy()

        # initialize covariance matrices
        covariance = np.cov(X.T) 

        covariances = np.array([covariance.copy() for _ in range(k)])

        # initialize equal mixing weights
        weights = np.ones(k) / k

        return means, covariances, weights

    def e_step(self, X, means, covariances, weights, k):
        # E-step: calculate the responsibilities
        responsibilities = np.zeros((X.shape[0], k))
        for i in range(k):
            numerator = (multivariate_normal.pdf(X, mean=means[i], cov=covariances[i]) * weights[i])
            responsibilities[:, i] = numerator

        # normalize the responsibilities
        responsibilities /= (responsibilities.sum(axis=1, keepdims=True) + 1e-10)

        return responsibilities

    def m_step(self, X, responsibilities, k):
        # M-step: update the parameters
        total_resp = np.sum(responsibilities, axis=0)

        # update weights
        weights = total_resp / X.shape[0]

        # update means
        means = (responsibilities.T @ X) / total_resp[:, np.newaxis]

        # initialize empty covariances
        covariances = np.zeros((k, X.shape[1], X.shape[1]))

        # update covariance matrices
        for i in range(k):
            diff = X - means[i]
            weighted_diff = responsibilities[:, i][:, np.newaxis] * diff
            covariances[i] = (weighted_diff.T @ diff) / total_resp[i]

            # covariance regularization
            covariances[i] += 1e-6 * np.eye(X.shape[1])

        return means, covariances, weights


    