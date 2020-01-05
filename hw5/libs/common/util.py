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

def save_to_csv(fname, data):
    csv_path = 'hw5/dataset/saved_csv/'
    np.savetxt(csv_path + fname, data, delimiter=',')  # X is an array

def plot_digit_data(data, fname):
    fig, numplt = plt.subplots(3, 3, subplot_kw=dict(xticks=[], yticks=[]))
    for i, axis in enumerate(numplt.flat):
        axis.imshow(data[i].reshape(28, 28), cmap='binary')
    plt.show()
    fig.savefig('hw5/results/%s.png' % fname, dpi=fig.dpi)
