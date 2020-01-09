from builtins import print

import numpy as np
import matplotlib.pyplot as plt

def int_to_tuple(Ks):
    lst = []
    for i in range(Ks):
        lst.append((i+1))
    return tuple(lst)

def step_size_to_tuple(Ks, step_init, step_size):
    lst = [step_init]
    init = step_init
    for i in range(Ks):
        init += step_size
        lst.append(round(init, 2))
    return tuple(lst)

def save_to_csv(fname, data, rot_path=None):
    if rot_path is not None: # used for Homework 6
        csv_path = rot_path + '/dataset/saved_csv/'
    else:
        csv_path = 'hw5/dataset/saved_csv/'
    np.savetxt(csv_path + fname, data, delimiter=',')  # X is an array

def plot_digit_data(data, fname, rot_path=None):
    fig, numplt = plt.subplots(3, 3, subplot_kw=dict(xticks=[], yticks=[]))
    for i, axis in enumerate(numplt.flat):
        axis.imshow(data[i].reshape(28, 28), cmap='binary')
    plt.show()
    if rot_path is not None:
        fig.savefig(rot_path+'/results/%s.png' % fname, dpi=fig.dpi)
    else:
        fig.savefig('hw5/results/%s.png' % fname, dpi=fig.dpi)

def get_unique_list(my_list):
    return list(set(my_list))

# to make it easier to analyze, take only digit={0, 1, ...}
def filter_dataset(selected_digits, X_train, Y_train):
    new_X_train = []  # digit = {0, 1, ..}
    new_Y_train = []
    for i in range(len(Y_train)):
        # if int(Y_train[i]) == 0 or int(Y_train[i]) == 1:
        if int(Y_train[i]) in selected_digits:
            new_X_train.append(X_train[i])
            new_Y_train.append(Y_train[i])
    return new_X_train, new_Y_train

# Available Digit = {0, 1, 2, ..., 10}; MAX(n_clusters) = 10
def get_selected_digits(n_clusters):
    digits = []
    for i in range(n_clusters):
        digits.append(i)
    return digits