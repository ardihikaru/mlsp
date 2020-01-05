'''
Source: https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
Step by step:
1. Split the training data by label.
2. For each set, fit a KDE to obtain a generative model of the data.
   This allows you for any observation x and label y to compute a likelihood P(x | y).
3. From the number of examples of each class in the training set, compute the class prior, P(y).
4. For an unknown point x, the posterior probability for each class is P(y | x)∝P(x | y)P(y).
   The class which maximizes this posterior is the label assigned to the point.
'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import numpy as np

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE

    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes]
        self.models = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self

    def __predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models]).T
        result = np.exp(logprobs + self.logpriors)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes[np.argmax(self.__predict_proba(X), 1)]

    def eval_acc(self, prob, Y_test):
        accuracy = np.sum(prob == Y_test).astype(float) / len(Y_test)
        return accuracy
