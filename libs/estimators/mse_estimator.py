import numpy as np

class MSE():
    def calc(self, yHat, y):
        # return ((yHat - y) ** 2) / 2
        return np.sum((yHat - y) ** 2) / y.size
