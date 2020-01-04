# library: pip install KDEpy
# Source: https://github.com/tommyod/KDEpy
# Docs: https://kdepy.readthedocs.io/en/latest/

from sklearn.decomposition import PCA
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
from scipy.stats import norm

from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple, save_to_csv
from hw5.libs.algo.knn_theano import KNNTheano
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

#%%
def plot_digit_data(data):
    fig, numplt = plt.subplots(3, 3, subplot_kw=dict(xticks=[], yticks=[]))
    for i, axis in enumerate(numplt.flat):
        axis.imshow(data[i].reshape(28, 28), cmap='binary')
    plt.show()
    fig.savefig('hw5/results/test_kde_plot_digits.png', dpi=fig.dpi)

def index_digit_ones(Y_train):
    idx_ones = []
    for i in range(len(Y_train)):
        if int(Y_train[i]) == 1:
            idx_ones.append(i)
    return idx_ones

if __name__ == '__main__':
    # dataset = Dataset(train_data=1000, test_data=100)
    # dataset = Dataset(train_data=40, test_data=10)
    dataset = Dataset(train_data=80, test_data=20)
    # dataset = Dataset()
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()

    print("sebelum PCA = ", X_train.shape)

    # Dimensional reduction
    # pca = PCA(n_components=2, whiten=False)
    pca = PCA(n_components=64, whiten=False)
    X_train = pca.fit_transform(X_train)
    print("setelah PCA = ", X_train.shape)

    # print(X_train[1].shape)
    print(Y_train)

    idx_ones = index_digit_ones(Y_train)
    print("> idx_ones = ", idx_ones)

    # data = norm(loc=0, scale=1).rvs(2 ** 3)
    idxs = [12, 3, 6, 14, 23, 24, 40, 59, 67]
    data = []
    # estimator = []
    # x, y = [], []
    for idx in idxs:
        data.append(X_train[idx])
    # data = X_train[3]
    # data2 = X_train[6]
    # data3 = X_train[12]
    # data4 = X_train[8]
    # print("Plotting Label = ", Y_train[3], Y_train[6], Y_train[12])

    fig = plt.figure()
    # more styles: https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
    line_styles = ['--', '-', ':', ':', '-', ':', ':', ':', ':']
    for i in range(len(idxs)):
        estimator = FFTKDE(kernel='gaussian', bw='silverman')
        # x[i], y[i] = estimator[i].fit(data[i], weights=None).evaluate()
        x, y = estimator.fit(data[i], weights=None).evaluate()

        # plt.plot(x[i], y[i], label='Digit='+str(Y_train[idxs[i]]))
        plt.plot(x, y, linestyle=line_styles[i], label='IDX='+str(idxs[i])+'; Digit='+str(Y_train[idxs[i]]))

    plt.legend()
    plt.show()
    fig.savefig('hw5/results/test_kde.png', dpi=fig.dpi)

    # estimator = FFTKDE(kernel='gaussian', bw='silverman')
    # x, y = estimator.fit(data, weights=None).evaluate()
    # estimator2 = FFTKDE(kernel='gaussian', bw='silverman')
    # x2, y2 = estimator2.fit(data2, weights=None).evaluate()
    # estimator3 = FFTKDE(kernel='gaussian', bw='silverman')
    # x3, y3 = estimator3.fit(data3, weights=None).evaluate()
    # estimator4 = FFTKDE(kernel='gaussian', bw='silverman')
    # x4, y4 = estimator4.fit(data4, weights=None).evaluate()

    # fig = plt.figure()
    # plt.plot(x, y, label='Digit='+str(Y_train[3]))
    # plt.plot(x2, y2, label='Digit='+str(Y_train[6]))
    # plt.plot(x3, y3, label='Digit='+str(Y_train[12]))
    # plt.plot(x4, y4, label='Digit='+str(Y_train[8]))
    # plt.legend()
    # plt.show()
    # fig.savefig('hw5/results/test_kde.png', dpi=fig.dpi)

    # new_data = [data, data2, data3, data4]
    # new_data = data
    new_data = pca.inverse_transform(data)

    plot_digit_data(new_data)
    # print(new_data.shape)
    # print("48 new data points generated : ")
    # print()



