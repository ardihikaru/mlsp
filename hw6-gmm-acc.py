from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple, save_to_csv, get_unique_list, filter_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from datetime import datetime
from hw6.libs.algo.gmm import MyGMM

if __name__ == '__main__':
    # dataset = Dataset(train_data=1000, test_data=100)
    # dataset = Dataset(train_data=20, test_data=10)
    # dataset = Dataset(train_data=40, test_data=10)
    # dataset = Dataset(train_data=80, test_data=20)
    dataset = Dataset(train_data=800, test_data=200)
    # dataset = Dataset()
    X_train, Y_train, _, _ = dataset.get_dataset()
    acc_scores = []

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
    gmm = MyGMM(n_clusters=len(unique_labels)) # n_clusters = total number of unique digits (labels)

    '''
    Do dimension Reduction first, and analyze the result
    Default: DISABLED; you may enable this. 
    RESULT: In clustering, PCA does not affect the accuracy!
    '''
    # pca_x_train = PCA(n_components=2, whiten=False)
    # X_train = pca_x_train.fit_transform(X_train)

    t0 = datetime.now()
    # Start GMM: Sklearn
    highest_acc = 0.0
    for i in range(K):
        gmm.fit(X_train)
        y_gmm = gmm.predict(X_train)
        accuracy = gmm.eval_acc(y_gmm, Y_train)
        acc_scores.append(accuracy)
        highest_acc = accuracy if accuracy > highest_acc else highest_acc

    elapsed_time = datetime.now() - t0

    fig = plt.figure()
    mean_acc = str(round(np.mean(np.array(acc_scores)), 2))
    title = "Highest (Red) = " + str(round(highest_acc, 2)) + "; AVG (Black) = " + mean_acc
    plt.title(title)
    plt.plot(ks, acc_scores, label='accuracy')
    plt.axhline(highest_acc, color='red', linestyle='dashed', linewidth=2)
    plt.axhline(np.mean(np.array(acc_scores)), color='k', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.show()
    fig.savefig('hw6/results/result-gmm-accuracy.png', dpi=fig.dpi)

    save_to_csv('gmm-acc.csv', acc_scores, "hw6")
    save_to_csv('gmm-exec-time.csv', [elapsed_time.total_seconds()], "hw6")
