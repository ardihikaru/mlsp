import theano, csv
import numpy as np
from theano import tensor as T

class KNNTheano():
    def __init__(self, k):
        # Create theano functions
        self.nfcn = self.__kneighbors(self.__edistance)
        self.gfcn = self.__get_nearest()
        self.K = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    # def score(self, X, Y):
    def score(self, X_test, Y_test):
        correct = 0
        for idx in range(len(X_test)):
            l = self.nfcn(self.X_train, self.Y_train, X_test[idx], self.K)
            m = self.gfcn(l)  # predicted label
            g = Y_test[idx]  # real label
            # estimate the results
            if m == g:
                correct = correct + 1
            # print(" ## idx= ", idx, "; Y_test[idx] = ", Y_test[idx], " CORRECT ? ", (m == g), m, g, l)
        score = (correct / float(len(X_test))) * 100.0
        return score

    def __edistance(self, x1, x2):
        d = x1 - x2
        z = T.sqrt(T.dot(d, d))
        return z

    # returns the labels of K neighbors
    def __kneighbors(self, dist):
        print("> returns the labels of K neighbors")
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
    def __get_nearest(self):
        l = T.ivector("l")
        z = T.extra_ops.bincount(l)
        a = T.argmax(z)
        o = l[a]
        return theano.function(inputs=[l], outputs=o)
