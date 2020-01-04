# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# Code: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/supervised_class/knn.py

import matplotlib.pyplot as plt
from hw5.libs.algo.knn import KNN
from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple
from datetime import datetime



if __name__ == '__main__':
    dataset = Dataset(max_data=20, train_percent=0.8)
    Xtrain, Ytrain, Xtest, Ytest = dataset.get_dataset()
    # print("total train = ", len(Xtrain))
    # print("total Xtest = ", len(Xtest))

    print(Ytest)

    train_scores = []
    test_scores = []

    # Define number of iteration (K)
    K = 5
    ks = int_to_tuple(K) # used to plot the results

    # Start KNN
    for i in range(K):
        k = i+1
        print("\nk =", k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))

        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    fig = plt.figure()
    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()
    fig.savefig('hw5/results/result.png', dpi=fig.dpi)
