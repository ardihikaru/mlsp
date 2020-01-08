from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple, save_to_csv, get_unique_list, filter_dataset, get_selected_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from hw6.libs.algo.gmm import MyGMM

if __name__ == '__main__':
    # dataset = Dataset(train_data=1000, test_data=100)
    # dataset = Dataset(train_data=20, test_data=10)
    # dataset = Dataset(train_data=40, test_data=10)
    # dataset = Dataset(train_data=80, test_data=20)
    dataset = Dataset(train_data=800, test_data=200)
    # dataset = Dataset()
    # X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    init_X_train, init_Y_train, _, _ = dataset.get_dataset()
    acc_scores = [] # ONLY collect the highest accuracy!

    # Simulate clustering in K times.
    K = 50
    # K = 5
    ks = int_to_tuple(K) # used to plot the results

    # simulate different number of clusters
    n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10] # from n=2 ~ n=10 (max)
    # n_clusters = [2, 3] # from n=2 ~ n=10 (max)

    # Start simulation ...
    gmm, reduced_X_train, y_gmm, unique_labels = None, None, None, None
    for i in range(len(n_clusters)):
        selected_digits = get_selected_digits(n_clusters[i])
        X_train, Y_train = filter_dataset(selected_digits, init_X_train, init_Y_train)

        # Used for visualization only
        pca = PCA(n_components=2, whiten=False)
        reduced_X_train = pca.fit_transform(X_train)

        unique_labels = get_unique_list(Y_train)
        gmm = MyGMM(len(unique_labels))  # n_clusters = total number of unique digits (labels)

        # Start gmm: Sklearn
        highest_acc = 0.0
        for j in range(K):
            # gmm.fit(X_train)
            gmm.fit(reduced_X_train)
            # y_gmm = gmm.predict(X_train)
            y_gmm = gmm.predict(reduced_X_train)
            accuracy = gmm.eval_acc(y_gmm, Y_train) * 100
            # acc_scores.append(accuracy)
            highest_acc = accuracy if accuracy > highest_acc else highest_acc

        str_hacc = str(round(highest_acc, 2))
        print(" >>> highest_acc of n_clusters[%s] = %s " % (str(n_clusters[i]), str_hacc))
        acc_scores.append(round(highest_acc, 2))

    fig = plt.figure()
    objects = tuple([("D=" + str(n)) for n in n_clusters])
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, acc_scores, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Digit (D)')
    plt.title('Eval. with diff. n_clusters; Dataset size = %s' % str(len(Y_train)))
    plt.show()
    fig.savefig('hw6/results/result-gmm-n_clusters.png', dpi=fig.dpi)

    # # Plt results
    gmm.visualize_predict_proba(reduced_X_train, Y_train, y_gmm, len(unique_labels), "result-gmm-x_clusters")
    # centers = gmm.cluster_centers_()
    # pca_center = PCA(n_components=2, whiten=False)
    # reduced_centers = pca_center.fit_transform(centers)
    # gmm.visualize(reduced_X_train, Y_train, y_gmm, reduced_centers, "result-gmm-x_clusters")
