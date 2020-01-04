import numpy as np

def int_to_tuple(Ks):
    lst = []
    for i in range(Ks):
        lst.append((i+1))
    return tuple(lst)

def save_to_csv(fname, data):
    csv_path = 'hw5/dataset/saved_csv/'
    np.savetxt(csv_path + 'exec-knn-theano.csv', data, delimiter=',')  # X is an array