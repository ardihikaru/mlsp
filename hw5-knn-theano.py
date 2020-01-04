import theano, csv
import numpy as np

from theano import tensor as T
from random import randint, shuffle

from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple
from hw5.libs.algo.knn_sklearn import KNNSkLearn
from datetime import datetime
import matplotlib.pyplot as plt

def edistance(x1, x2):
    d = x1 - x2
    z = T.sqrt(T.dot(d, d))
    return z

# returns the labels of K neighbors
def kneighbors(dist):
    train  = T.matrix("train")
    labels = T.ivector("labels")
    test_sample = T.vector("sample")
    k = T.iscalar("k")
    l = T.ivector("l")

    r, u = theano.scan(lambda row: dist(row, test_sample), sequences=train)
    z = T.argsort(r)
    i = z[0:k]
    l = labels[i]
    return theano.function(inputs=[train, labels, test_sample, k], outputs=l)

# Returns a label by voting nearest point labels
def get_nearest():
    l = T.ivector("l")
    z = T.extra_ops.bincount(l)
    a = T.argmax(z)
    o = l[a]
    return theano.function(inputs=[l], outputs=o)


if __name__ == '__main__':
    dataset = Dataset(train_data=80, test_data=20)
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    train_scores = []
    test_scores = []

    # Define number of iteration (K)
    K = 20
    ks = int_to_tuple(K)  # used to plot the results

# def main():
#     k = 5
#     train, labels, labels_map = load_data('iris.data')

    # Create theano functions
    nfcn = kneighbors(edistance)
    gfcn = get_nearest()

    n = len(Y_train)
    n = randint(n / 2, n)

    print("split len = ", n)
    # print("labels map len = ", len(labels_map))
    print("data-set len = ", len(X_train))
    print("K = ", K)

    # Split train data
    t1 = X_train[0:n-1]
    t2 = X_train[n:]
    l1 = Y_train[0:n-1]
    l2 = Y_train[n:]
    correct = 0

    for idx in range(len(t2)):
        l = nfcn(t1, l1, t2[idx], K)
        m = gfcn(l)
        g = l2[idx]
        # estimate the results
        if m == g:
            correct = correct + 1

    p = (correct / float(len(t2))) * 100.0
    print("test results:", p, "%")

