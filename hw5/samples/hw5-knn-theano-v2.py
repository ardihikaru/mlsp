'''
Super fast k-nearest-neighbour search for GPU
Based on the method used in "Learning To Remember Rare Events"
by Lukasz Kaiser, Ofir Nachun, Aurko Roy, and Samy Bengio
Paper: https://openreview.net/pdf?id=SJTQLdqlg
'''

import numpy as np
import theano
import theano.tensor as T
import time
import os

from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple

if __name__ == '__main__':
    dataset = Dataset(train_data=10, test_data=3)
    # dataset = Dataset()
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    train_scores = []
    test_scores = []

    print(" > Total X_train = ", len(X_train))
    print(" > Total X_test = ", len(X_test))

    print(">>> Y_test = ", Y_test)

    # Define number of iteration (K)
    K = 20
    ks = int_to_tuple(K)  # used to plot the results


    def l2_normalize(x, dim, epsilon=1e-12):
        square_sum = T.sum(T.sqr(x), axis=dim)
        x_inv_norm = T.true_div(1, T.sqrt(T.maximum(square_sum, epsilon)))
        x_inv_norm = x_inv_norm.dimshuffle(0, 'x')
        return T.mul(x, x_inv_norm)


    ### Theano KNN ###
    # Construct the theano graph
    x_keys = T.matrix('x_keys')
    x_queries = T.matrix('x_keys')

    print(">>> x_keys = ", x_keys)
    print(">>> x_queries = ", x_queries)

    # dim = 1  # ini buat apa ya?

    normalized_keys = l2_normalize(x_keys, dim=1)
    normalized_query = l2_normalize(x_queries, dim=1)
    # normalized_keys = l2_normalize(x_keys, dim=dim)
    # normalized_query = l2_normalize(x_queries, dim=dim)
    query_result = T.dot(normalized_keys, normalized_query.T)
    pred = T.argmax(query_result, axis=0)

    print(">>> normalized_keys = ", normalized_keys)
    print(">>> normalized_query = ", normalized_query)
    print(">>> query_result = ", query_result)
    print(">>> pred = ", pred)

    # Declare the knn function
    knn = theano.function(inputs=[x_keys, x_queries], outputs=pred)

    print("Evaluating theano KNN")
    start = time.time()
    nn_index = knn(X_train, X_test)
    print(">>> nn_index = ", nn_index)
    end = time.time()

    accuracy = np.sum(Y_train[nn_index] == Y_test).astype(float) / len(Y_test)

    print("Accuracy: " + str(accuracy * 100) + '%')
    print('Took', end - start, 'seconds')