import matplotlib.pyplot as plt
from hw5.libs.common.util import int_to_tuple
import csv

def read_data(fname):
    data_path = "hw5/dataset/saved_csv/"
    fpath = data_path + fname

    with open(fpath, 'r') as f:
        reader = csv.reader(f)
        # dataset = list(reader)
        return [float(line[0]) for line in list(reader)]
        # print(" >> dataset = ", dataset)
        # print(" >> LEN dataset = ", len(dataset))
        # print(" >> data = ", data)
        # print(" >> dataset = ", reader)

if __name__ == '__main__':
    # Define number of iteration (K)
    K = 50
    ks = int_to_tuple(K)  # used to plot the results

    scratch = read_data('exec-knn-scratch.csv')
    sklearn = read_data('exec-knn-sklearn.csv')
    theano = read_data('exec-knn-theano.csv')
    theano_colab = read_data('exec-knn-theano-google_colab.csv')

    print(" total len = ", len(scratch), len(sklearn), len(theano))

    fig = plt.figure()
    plt.plot(ks, scratch, label='scratch exec. time')
    plt.plot(ks, sklearn, label='sklearn exec. time')
    plt.plot(ks, theano, label='theano exec. time')
    plt.plot(ks, theano_colab, label='theano G-Colab exec. time')
    plt.legend()
    plt.show()
    fig.savefig('hw5/results/result-knn-compare-exec-times.png', dpi=fig.dpi)
