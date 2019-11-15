import numpy as np
from hw1.libs.distributions.gamma_dist import GammaDist

class MAP():
    def __init__(self, ext_dataset=None, **input_variables):
        if ext_dataset is not None:
            self.data = ext_dataset
        else:
            self.data = GammaDist(**input_variables) # default: Gamma Dataset

    def __inits_variables(self):
        self.n = self.data.get_size()
        self.x = self.data.get_dataset()
        self.x_sum = np.sum(self.x)
        self.alpha = self.data.get_alpha()
        self.beta = self.data.get_beta()

    def run(self):
        self.__inits_variables()
        self.__calc_yhat()

    def multiple_run(self, n_retry, new_alpha=None, data=None):
        self.array_yhat = np.empty(shape=[n_retry])
        for i in range(0, n_retry):
            if data is not None: # data regeneration is performed before assigned here
                self.data = data
            else:
                self.data.regenerate_dataset(new_alpha)
            self.run()
            self.array_yhat[i] = self.get_yhat()

    def get_array_yhat(self):
        return self.array_yhat

    def __calc_yhat(self):
        self.yhat = (self.n + self.alpha - 1) / (self.x_sum + self.beta)

    def get_yhat(self):
        return self.yhat

    def plot(self):
        self.data.plot_dataset()