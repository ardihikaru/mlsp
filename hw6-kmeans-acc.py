from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple, save_to_csv, get_unique_list, filter_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

# from sklearn.cluster import KMeans
from hw6.libs.algo.kmeans import MyKMeans

if __name__ == '__main__':
    # dataset = Dataset(train_data=1000, test_data=100)
    # dataset = Dataset(train_data=20, test_data=10)
    # dataset = Dataset(train_data=40, test_data=10)
    # dataset = Dataset(train_data=80, test_data=20)
    dataset = Dataset(train_data=800, test_data=200)
    # dataset = Dataset()
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    acc_scores = []
    # stored_accuracy = [0, 0]  # [<acc>, <k-th>

    # Simulate clustering in K times.
    K = 50
    # K = 5
    ks = int_to_tuple(K) # used to plot the results

    # to make it easier to analyze, take only digit={0, 1, ...}
    selected_digits = [1, 2, 3]
    # selected_digits = [1, 2]
    X_train, Y_train = filter_dataset(selected_digits, X_train, Y_train)

    # Used for visualization only
    pca = PCA(n_components=2, whiten=False)
    reduced_X_train = pca.fit_transform(X_train)

    unique_labels = get_unique_list(Y_train)
    kmeans = MyKMeans(n_clusters=len(unique_labels)) # n_clusters = total number of unique digits (labels)

    '''
    Do dimension Reduction first, and analyze the result
    Default: DISABLED; you may enable this. 
    RESULT: In clustering, PCA does not affect the accuracy!
    '''
    # pca_x_train = PCA(n_components=64, whiten=False)
    # X_train = pca_x_train.fit_transform(X_train)

    # Start KMeans: Sklearn
    highest_acc = 0.0
    for i in range(K):
        kmeans.fit(X_train)
        y_kmeans = kmeans.predict(X_train)
        accuracy = kmeans.eval_acc(y_kmeans, Y_train)
        acc_scores.append(accuracy)
        highest_acc = accuracy if accuracy > highest_acc else highest_acc

    fig = plt.figure()
    mean_acc = str(round(np.mean(np.array(acc_scores)), 2))
    title = "Highest (Red) = " + str(round(highest_acc, 2)) + "; AVG (Black) = " + mean_acc
    plt.title(title)
    plt.plot(ks, acc_scores, label='accuracy')
    plt.axhline(highest_acc, color='red', linestyle='dashed', linewidth=2)
    plt.axhline(np.mean(np.array(acc_scores)), color='k', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.show()
    fig.savefig('hw6/results/result-kmeans-accuracy.png', dpi=fig.dpi)

    # Sample without looping.
    # kmeans.fit(X_train)
    # y_kmeans = kmeans.predict(X_train)
    # accuracy = kmeans.eval_acc(y_kmeans, Y_train)
    # print(" >> accuracy = ", accuracy)
    #
    # # Plt results
    # centers = kmeans.cluster_centers_()
    # pca_center = PCA(n_components=2, whiten=False)
    # reduced_centers = pca_center.fit_transform(centers)
    # kmeans.visualize(reduced_X_train, Y_train, y_kmeans, reduced_centers)

