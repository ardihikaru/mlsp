import numpy as np
from libs.distributions.exp_dist import ExpDist

class MLE():
    def __init__(self, **input_variables):
        self.exp_dist = ExpDist(**input_variables)

    def run(self):
        self.__calc_yhat()

    def multiple_run(self, n_retry):
        self.array_yhat = np.empty(shape=[n_retry])
        for i in range(0, n_retry):
            self.exp_dist.regenerate_dataset()
            self.run()
            self.array_yhat[i] = self.get_yhat()

    def get_array_yhat(self):
        return self.array_yhat

    def __calc_yhat(self):
        self.yhat = 1 / np.average(self.exp_dist.get_dataset())

    def get_yhat(self):
        return self.yhat

    def plot(self):
        self.exp_dist.plot_dataset()
