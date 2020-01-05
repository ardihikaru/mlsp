'''
####### Try calculating optimal bandwidth ########
## Selecting the bandwidth via cross-validation ##
##################################################

Summary:
# Too wide a bandwidth leads to a high-bias estimate -> under-fitting
# too narrow a bandwidth leads to a high-variance estimate -> over-fitting
# KernelDensity = estimator in Scikit-Learn
# GridSearchCV = to optimize the bandwidth for the preceding dataset
# small dataset = use leave-one-out cross-validation

How does it works:
1.  KernelDensity() is simply estimating the density of the given data
    - it places a Gaussian on each data point and then sum up all of these Gaussians
    - When normalizing by the number of data points, this should yield the PDF.
2.  GridSearchCV is an algorithm that selects the optimal bandwidth of the Gaussians
    that KernelDensity() is going to use.
    - To achieve this, GridSearchCV tries out a certain bandwidth, and
    - lets KernelDensity() estimate the pdf using K-1 folds
    - It then tests how good the KDE is on the last fold by computing the log-likelihood.
    - NB: GridSearchCV does this K times and averages the likelihood.
'''
from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import plot_digit_data
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = Dataset(train_data=80, test_data=20)
    # dataset = Dataset(train_data=800, test_data=200)
    # dataset = Dataset()
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()

    # Dimensional reduction: Enable this to get faster result, but it will affect further process.
    # pca = PCA(n_components=64, whiten=False)
    pca = PCA(n_components=80, whiten=False)
    X_train = pca.fit_transform(X_train)

    # 1. Sample: Only Capture digit=1;
    new_x = []
    for i in range(len(Y_train)):
        if int(Y_train[i]) == 1:
            # print("i=", i, "; label = ", Y_train[i])
            new_x.append(X_train[i])
    new_x = np.asarray(new_x)
    print(new_x.shape)

    bw = []
    ks = []
    for i in range(len(new_x)):
        x = new_x[i]
        '''
        cv : int, cross-validation generator or an iterable, optional
                Determines the cross-validation splitting strategy.
                Possible inputs for cv are:
        '''
        loo = LeaveOneOut()
        cv_loo = loo.get_n_splits(x)

        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=cv_loo)
        grid.fit(x[:, None])

        print("BEST GRID = ", grid.best_params_)
        bw.append(float(grid.best_params_["bandwidth"]))
        ks.append(i)
    print("AVG Bandwidth = ", np.average(bw))

    fig = plt.figure()
    plt.plot(ks, bw, label='optimal bandwidth')
    # plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()
    fig.savefig('hw5/results/kde-optimal-bandwidth.png', dpi=fig.dpi)

    new_data = pca.inverse_transform(new_x)
    plot_digit_data(new_data, 'kde_digits_opt.bw')



