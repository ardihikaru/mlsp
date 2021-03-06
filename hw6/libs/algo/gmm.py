import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class MyGMM:
    # n_clusters = total number of collected digit labels
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.gmm = GaussianMixture(n_components=n_clusters)

    # def fit(self, X):
    def fit(self, X):
        self.gmm.fit(X)

    # Generates variable "prob"
    def predict(self, X):
        return self.gmm.predict(X)

    def eval_acc(self, prob, Y_test):
        accuracy = np.sum(prob == Y_test).astype(float) / len(Y_test)
        return accuracy

    def visualize_predict_proba(self, reduced_X_train, Y_train, labels, n_clusters, fname="result-gmm"):
        fig = plt.figure()
        plt.scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=Y_train, s=100, cmap='inferno')
        plt.title('Original Distribution; n_clusters=%s' % n_clusters)
        plt.show()
        fig.savefig('hw6/results/result-gmm-original.png', dpi=fig.dpi)

        fig = plt.figure()
        probs = self.gmm.predict_proba(reduced_X_train)
        size = 50 * probs.max(1) ** 2  # square emphasizes differences
        plt.scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=labels, cmap='viridis', s=size)
        plt.title('Distribution generated by GMM; n_clusters=%s' % n_clusters)
        plt.show()
        fig.savefig('hw6/results/%s.png' % fname, dpi=fig.dpi)

    # def visualize(self, reduced_X_train, new_Y_train, y_kmeans, reduced_centers, fname="result-gmm"):
    #
    #     fig = plt.figure()
    #     plt.scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=new_Y_train, s=100, cmap='inferno')
    #     plt.show()
    #     fig.savefig('hw6/results/result-gmm-original.png', dpi=fig.dpi)
    #
    #     fig = plt.figure()
    #     plt.scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], s=100)
    #     plt.scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=y_kmeans, s=50, cmap='viridis')
    #     plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='red', s=200, alpha=0.5)
    #     plt.show()
    #     fig.savefig('hw6/results/%s.png' % fname, dpi=fig.dpi)
