import numpy as np
from libs.distributions.gamma_dist import GammaDist

class MAP():
    def __init__(self, **input_variables):
        self.gamma_dist = GammaDist(**input_variables)

    def __inits_variables(self):
        self.n = self.gamma_dist.get_size()
        self.x = self.gamma_dist.get_dataset()
        self.x_sum = np.sum(self.x)
        self.alpha = self.gamma_dist.get_alpha()
        self.beta = self.gamma_dist.get_beta()

    def run(self):
        self.__inits_variables()
        self.__calc_yhat()

    def multiple_run(self, n_retry, new_alpha=None):
        self.array_yhat = np.empty(shape=[n_retry])
        for i in range(0, n_retry):
            self.gamma_dist.regenerate_dataset(new_alpha)
            self.run()
            self.array_yhat[i] = self.get_yhat()

    def get_array_yhat(self):
        return self.array_yhat

    def __calc_yhat(self):
        self.yhat = (self.n + self.alpha - 1) / (self.x_sum + self.beta)

    def get_yhat(self):
        return self.yhat

    def plot(self):
        self.gamma_dist.plot_dataset()