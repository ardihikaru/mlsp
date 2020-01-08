from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import save_to_csv, get_unique_list
from hw6.libs.algo.kmeans import MyKMeans
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if __name__ == '__main__':
    train_scores = []
    test_scores = []
    exec_times = []
    best_test_score = 0
    stored_accuracy = [0, 0] # [<acc>, <k-th>

    # Disable this PCA to obtain the best result!
    # Evaluate: Try performing PCA first, and check the results
    # Result: Reducing PCA is not recommended, since it will dramatically reduce the accuracy.
    # Dimensional reduction
    # pca = PCA(n_components=16, whiten=False)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.fit_transform(X_test)

    # Define value of clustering simulation
    # K = 9
    # K = 1 # total number of simulations
    K = 4 # total number of simulations
    dataset_size = 200 # initialize started dataset size here
    # dataset_step_size = 100 # e.g. datasets = {100, 200, 300, ..., 1000}
    dataset_step_size = 200 # e.g. datasets = {100, 200, 300, ..., 1000}
    ratio_train, ratio_test = 0.8, 0.2 # should be total value=1 ! e.g. ratio = { (80,20), (160,40), ..., (800,200)  }

    datasets = []
    ds = [] # list of dataset size in each iterations
    for i in range(K):
        # print(" >> dataset_size = ", dataset_size)
        train_data = int(dataset_size * ratio_train)
        test_data = dataset_size - train_data
        dataset = Dataset(train_data=train_data, test_data=test_data)
        X_train, Y_train, X_test, Y_test = dataset.get_dataset()
        datasets.append({
            "X_train": X_train,
            "Y_train": Y_train,
            "X_test": X_test,
            "Y_test": X_train
        })
        ds.append(dataset_size)
        dataset_size += dataset_step_size
        # print(" >> train_data, test_data = ", train_data, test_data)

    # print(" >> ds = ", ds)

    # Start KMeans: sklearn
    print("Evaluating sklearn KMeans")
    for i in range(K):
        X_train = datasets[i]["X_train"]
        Y_train = datasets[i]["Y_train"]
        X_test = datasets[i]["X_test"]
        Y_test = datasets[i]["Y_test"]
        print("dataset_size = ", dataset_size, "; n_clusters = ", len(Y_train)) # n_clusters = len(Y_train)
        kmeans = MyKMeans(len(Y_train))
        kmeans.fit(X_train)

        t0 = datetime.now()
        train_prob = kmeans.predict(X_train)
        train_score = kmeans.eval_acc(train_prob, Y_train)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))

        t0 = datetime.now()
        test_prob = kmeans.predict(X_test)
        test_score = kmeans.eval_acc(test_prob, Y_test)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))
        exec_times.append((datetime.now() - t0).total_seconds())

        # update best accuracy (for comparison purpose later)
        if test_score > best_test_score:
            best_test_score = test_score
            stored_accuracy = [test_score, ds[i]]

    fig = plt.figure()
    plt.plot(ds, train_scores, label='train scores')
    plt.plot(ds, test_scores, label='test scores')
    plt.legend()
    plt.show()
    fig.savefig('hw6/results/result-kmeans.png', dpi=fig.dpi)
    save_to_csv('exec-kmeans.csv', exec_times, rot_path="..")
    save_to_csv('test-scores-kmeans.csv', test_scores, rot_path="..")
    save_to_csv('best-acc-kmeans.csv', stored_accuracy, rot_path="..")
