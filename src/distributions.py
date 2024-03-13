import numpy as np

class Gaussian():
    # Multivariate gaussian distribution

    def __init__(self, mean=np.zeros(2), cov=np.eye(2)):
        self.mean = mean
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        self.normalization = np.sqrt(np.linalg.det(self.inv_cov)) / (2*np.pi)

    def density(self, points):
        """
        Compute the density

        Parameters:
        Points: array of shape (n_points, 2)

        Return:
        density: array of shape (n_points)
        """
        centered = (points - self.mean)
        log_density = - 0.5 * np.sum(centered * (centered @ self.inv_cov), axis=1)
        return np.exp(log_density) * self.normalization


    def grad_log_density(self, points):
        """
        Compute the gradient of the log density

        Parameters:
        Points: array of shape (n_points, 2)

        Return:
        grad: array of shape (n_points, 2)
        """
        centered = (points - self.mean)
        grad = - centered @ self.inv_cov
        return grad
    

class MoG():
    # Mixture of multivariate gaussian distributions

    def __init__(self, means=[np.zeros(2)], covs=[np.eye(2)], weights=[1]):
        self.means = means
        self.covs = covs
        self.inv_covs = [np.linalg.inv(cov) for cov in covs]
        self.weights = weights

    def density(self, points):
        """
        Compute the unnormalized density

        Parameters:
        Points: array of shape (n_points, 2)

        Return:
        density: array of shape (n_points)
        """
        density = np.zeros(len(points))
        for mean, inv_cov, weight in zip(self.means, self.inv_covs, self.weights):
            centered = (points - mean)
            log_density = - 0.5 * np.sum(centered * (centered @ inv_cov), axis=1)
            density += np.exp(log_density) * weight

        return density


    def grad_log_density(self, points):
        """
        Compute the gradient of the log density

        Parameters:
        Points: array of shape (n_points, 2)

        Return:
        grad: array of shape (n_points, 2)
        """
        n, d = points.shape
        numerator = np.zeros((d, n))
        denominator = np.zeros(n)
        for mean, inv_cov, weigth in zip(self.means, self.inv_covs, self.weights):
            centered = (points - mean)
            dot = (centered @ inv_cov)
            log_density = - 0.5 * np.sum(centered * dot, axis=1)
            term = weigth * np.sqrt(np.linalg.det(inv_cov)) * np.exp(log_density)

            numerator += term * dot.T
            denominator += term
        return ( - numerator / denominator).T
    

class CircleDist:
    # Circular shape distribution, width radius R and thickness alpha
    def __init__(self, R=1, alpha=0.1):
        self.R = R
        self.alpha=alpha

    def density(self, points):
        """
        Compute the density

        Parameters:
        Points: array of shape (n_points, 2)

        Return:
        density: array of shape (n_points)
        """
        norms = np.linalg.norm(points, axis=1)
        log_density = - (norms - self.R)**2 / self.alpha**2
        return np.exp(log_density)


    def grad_log_density(self, points):
        """
        Compute the gradient of the log density

        Parameters:
        Points: array of shape (n_points, 2)

        Return:
        grad: array of shape (n_points, 2)
        """

        norms = np.linalg.norm(points, axis=1)
        grad = 2 * points.T * (self.R - norms) / norms
        return grad.T