from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNSkLearn():
    def __init__(self, k):
        self.neigh = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, Y):
        self.neigh.fit(X, Y)

    def score(self, X, Y):
        pred = self.neigh.predict(X)
        score = np.sum(pred == Y).astype(float) / len(Y)
        return score

