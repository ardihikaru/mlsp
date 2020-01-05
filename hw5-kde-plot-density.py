# library: pip install KDEpy
# Source: https://github.com/tommyod/KDEpy
# Docs: https://kdepy.readthedocs.io/en/latest/

'''
By performing PCA, we can analyze why reducing PCA component can reduce the accuracy of KDE calculation.
'''

from sklearn.decomposition import PCA
from KDEpy import FFTKDE

from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import plot_digit_data
import matplotlib.pyplot as plt

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

    print("before PCA = ", X_train.shape)
    # Dimensional reduction
    # pca = PCA(n_components=2, whiten=False)
    # pca = PCA(n_components=64, whiten=False)
    pca = PCA(n_components=80, whiten=False)
    X_train = pca.fit_transform(X_train)
    print("after PCA = ", X_train.shape)

    # print(X_train[1].shape)
    print(Y_train)

    idx_ones = index_digit_ones(Y_train)
    print("> idx_ones = ", idx_ones)

    idxs = [12, 3, 6, 14, 23, 24, 40, 59, 67] # this is extracted indices of digit=1; with idx=12 as digit=3
    data = []
    for idx in idxs:
        data.append(X_train[idx])

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
    fig.savefig('hw5/results/visualize_kde.png', dpi=fig.dpi)

    new_data = pca.inverse_transform(data)

    plot_digit_data(new_data, 'test_kde_plot_digits')



