# from math import exp
import numpy as np
# from .mse_estimator import MSE
from libs.distributions.exp_dist import ExpDist

class MLE():
    # def __init__(self, lam, x):
    def __init__(self, **expo_opt):
        # self.y = lam
        # self.x = x
        self.expo = ExpDist(**expo_opt)

    # def __calc_y(self, x, lam):
    #     return [lam * exp(-lam * xi) for xi in x]

    # def get_y(self, x, lam):
    #     return self.__calc_y(x, lam)
    def run(self):
        self.__calc_yhat()
        # self.__calc_mse()

    def multiple_run(self, n):
        self.array_yhat = np.empty(shape=[n])
        for i in range(1, n):
            self.run()
            self.array_yhat[i] = self.get_yhat()

    def get_array_yhat(self):
        return self.array_yhat

    # def get_y(self):
    #     return self.y

    def __calc_yhat(self):
        self.yhat = 1 / np.average(self.expo.get_dataset())

    def get_yhat(self):
        return self.yhat

    # def __calc_mse(self):
    #     self.mse = MSE().calc(self.yhat, self.y)
    #
    # def get_mse(self):
    #     return self.mse

