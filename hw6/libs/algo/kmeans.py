import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MyKMeans:
    # n_clusters = total number of collected digit labels
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)

    # def fit(self, X):
    def fit(self, X):
        self.kmeans.fit(X)

    # Generates variable "prob"
    def predict(self, X):
        return self.kmeans.predict(X)

    def cluster_centers_(self):
        return self.kmeans.cluster_centers_

    def eval_acc(self, prob, Y_test):
        accuracy = np.sum(prob == Y_test).astype(float) / len(Y_test)
        return accuracy

    def visualize(self, reduced_X_train, new_Y_train, y_kmeans, reduced_centers, fname="result-kmeans"):

        fig = plt.figure()
        plt.scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=new_Y_train, s=100, cmap='inferno')
        plt.show()
        fig.savefig('hw6/results/result-kmeans-original.png', dpi=fig.dpi)

        fig = plt.figure()
        plt.scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], s=100)
        plt.scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=y_kmeans, s=50, cmap='viridis')
        plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='red', s=200, alpha=0.5)
        plt.show()
        fig.savefig('hw6/results/%s.png' % fname, dpi=fig.dpi)
