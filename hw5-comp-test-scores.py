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

    scratch = read_data('knn-test-scores-scratch.csv')
    sklearn = read_data('knn-test-scores-sklearn.csv')
    theano = read_data('knn-test-scores-theano.csv')

    print(" total len = ", len(scratch), len(sklearn), len(theano))

    fig = plt.figure()
    plt.plot(ks, scratch, label='scratch test scores')
    plt.plot(ks, sklearn, label='sklearn test scores')
    plt.plot(ks, theano, label='theano test scores')
    plt.legend()
    plt.show()
    fig.savefig('hw5/results/result-knn-compare-test-scores.png', dpi=fig.dpi)
