# from math import exp
import numpy as np
# from .mse_estimator import MSE

class MLE():
    def __init__(self, lam, x):
        self.y = lam
        self.x = x
    # def __calc_y(self, x, lam):
    #     return [lam * exp(-lam * xi) for xi in x]

    # def get_y(self, x, lam):
    #     return self.__calc_y(x, lam)
    def run(self):
        self.__calc_yhat()
        # self.__calc_mse()

    def get_y(self):
        return self.y

    def __calc_yhat(self):
        self.yhat = 1 / np.average(self.x)

    def get_yhat(self):
        return self.yhat

    # def __calc_mse(self):
    #     self.mse = MSE().calc(self.yhat, self.y)
    #
    # def get_mse(self):
    #     return self.mse

