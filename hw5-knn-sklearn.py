from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple, save_to_csv
from hw5.libs.algo.knn_sklearn import KNNSkLearn
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = Dataset(train_data=80, test_data=20)
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    train_scores = []
    test_scores = []
    exec_times = []

    # Define number of iteration (K)
    K = 50
    ks = int_to_tuple(K)  # used to plot the results

    # Start KNN: sklearn
    print("Evaluating sklearn KNN")
    for i in range(K):
        k = i+1
        print("\nk =", k)
        knn = KNNSkLearn(k)
        t0 = datetime.now()
        knn.fit(X_train, Y_train)
        print("Training time:", (datetime.now() - t0))

        t0 = datetime.now()
        train_score = knn.score(X_train, Y_train)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))

        t0 = datetime.now()
        test_score = knn.score(X_test, Y_test)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))
        exec_times.append((datetime.now() - t0).total_seconds())

        # Result will be different in this way (below), since the training data will be trained only once.
        # it resulted same prediction and accuracy, though we changed number of neighboar (k)
        # neigh = KNeighborsClassifier(n_neighbors=K)
        #
        # t0 = datetime.now()
        # neigh.fit(X_train, Y_train)
        # print("Training time:", (datetime.now() - t0))
        #
        # t0 = datetime.now()
        # train_pred = neigh.predict(X_train)
        # train_score = np.sum(train_pred == Y_train).astype(float) / len(Y_train)
        # train_scores.append(train_score)
        # print("Train accuracy:", train_score)
        # print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))
        #
        # t0 = datetime.now()
        # test_pred = neigh.predict(X_test)
        # test_score = np.sum(test_pred == Y_test).astype(float) / len(Y_test)
        # test_scores.append(test_score)
        # print("Test accuracy:", train_score)
        # print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))

    fig = plt.figure()
    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()
    fig.savefig('hw5/results/result-knn-sklearn.png', dpi=fig.dpi)
    save_to_csv('exec-knn-sklearn.csv', exec_times) # X is an array


