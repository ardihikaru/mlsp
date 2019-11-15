import numpy as np
from hw1.libs.distributions.exp_dist import ExpDist

class MLE():
    def __init__(self, ext_dataset=None, **input_variables):
        if ext_dataset is not None:
            self.data = ext_dataset
        else:
            self.data = ExpDist(**input_variables) # default: Exponential Dataset

    def run(self):
        self.__calc_yhat()

    def multiple_run(self, n_retry, data=None):
        self.array_yhat = np.empty(shape=[n_retry])
        for i in range(0, n_retry):
            if data is not None: # data regeneration is performed before assigned here
                self.data = data
            else:
                self.data.regenerate_dataset()
            self.run()
            self.array_yhat[i] = self.get_yhat()

    def get_array_yhat(self):
        return self.array_yhat

    def __calc_yhat(self):
        self.yhat = 1 / np.average(self.data.get_dataset())

    def get_yhat(self):
        return self.yhat

    def plot(self):
        self.data.plot_dataset()
