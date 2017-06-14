import numpy as np
import scipy.stats as stats

class GaussSampler:
    def __init__(self, training_data):
        self.mean = np.mean(training_data, 0) #todo: Test dimension
        self.cov = np.cov(training_data)

    def random_sample(self, amount=1):
        return stats.multivariate_normal.rvs(mean=self.mean, cov=self.cov, amount=amount)

    def sample_with_mean(self, x, scale_cov=1, amount=1):
        return stats.multivariate_normal.rvs(mean=x, cov=self.cov*scale_cov, amount=amount)

    def permutation(self, x):
        y = self.random_sample()
        return y # TODO

