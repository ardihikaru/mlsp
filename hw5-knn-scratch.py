# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# Code: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/supervised_class/knn.py

import matplotlib.pyplot as plt
from hw5.libs.algo.knn import KNN
from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple, save_to_csv
from datetime import datetime

if __name__ == '__main__':
    dataset = Dataset(train_data=80, test_data=20)
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    train_scores = []
    test_scores = []
    exec_times = []
    best_test_score = 0
    stored_accuracy = [0, 0] # [<acc>, <k-th>]

    # Define number of iteration (K)
    K = 50
    ks = int_to_tuple(K) # used to plot the results

    # Start KNN: Scratch
    print("Evaluating Scratch KNN")
    for i in range(K):
        k = i+1
        print("\nk =", k)
        knn = KNN(k)
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

        # update best accuracy (for comparison purpose later)
        if test_score > best_test_score:
            best_test_score = test_score
            stored_accuracy = [test_score, k]

    fig = plt.figure()
    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()
    fig.savefig('hw5/results/result-knn-scratch.png', dpi=fig.dpi)
    save_to_csv('exec-knn-scratch.csv', exec_times) # X is an array
    save_to_csv('knn-test-scores-scratch.csv', test_scores)
    save_to_csv('knn-best-acc-scratch.csv', stored_accuracy)
