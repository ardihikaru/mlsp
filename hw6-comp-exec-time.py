import matplotlib.pyplot as plt
from hw5.libs.common.util import int_to_tuple
import csv
import numpy as np
import pandas as pd
import seaborn as sns

def read_data(fname):
    data_path = "hw6/dataset/saved_csv/"
    fpath = data_path + fname

    with open(fpath, 'r') as f:
        reader = csv.reader(f)
        return [float(line[0]) for line in list(reader)]

if __name__ == '__main__':
    kmeans = read_data('kmeans-exec-time_n=3_no-PCA.csv')[0]
    kmeans_pca = read_data('kmeans-exec-time_n=3_PCA=2.csv')[0]
    gmm = read_data('gmm-exec-time_n=3_no-PCA.csv')[0]
    gmm_pca = read_data('gmm-exec-time_n=3_PCA=2.csv')[0]

    sns.set_context('paper')

    d = {
        'method':
            ['kmeans', 'kmeans_pca', 'gmm', 'gmm_pca'],
        'exec_times':
            [kmeans, kmeans_pca, gmm, gmm_pca]
    }
    df = pd.DataFrame(data=d)

    fig = plt.figure()
    sns.barplot(x="method", y="exec_times", data=df)
    plt.show()
    fig.savefig('hw6/results/result-kmeans_gmm-compare-exec-time.png', dpi=fig.dpi)
