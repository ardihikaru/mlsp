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
    kmeans = read_data('kmeans-acc_n=3_no-PCA.csv')
    kmeans_pca = read_data('kmeans-acc_n=3_PCA=2.csv')
    gmm = read_data('gmm-acc_n=3_no-PCA.csv')
    gmm_pca = read_data('gmm-acc_n=3_PCA=2.csv')

    mean_kmeans = round(np.mean(np.array(kmeans)), 2)
    max_kmeans = round(np.max(np.array(kmeans)), 2)
    mean_kmeans_pca = round(np.mean(np.array(kmeans_pca)), 2)
    max_kmeans_pca = round(np.max(np.array(kmeans_pca)), 2)
    mean_gmm = round(np.mean(np.array(gmm)), 2)
    max_gmm = round(np.max(np.array(gmm)), 2)
    mean_gmm_pca = round(np.mean(np.array(gmm_pca)), 2)
    max_gmm_pca = round(np.max(np.array(gmm_pca)), 2)

    sns.set_context('paper')

    d = {
        'metric':
            ['mean', 'mean', 'mean', 'mean', 'max', 'max', 'max', 'max'],
        'method':
            ['kmeans', 'kmeans_pca', 'gmm', 'gmm_pca', 'kmeans', 'kmeans_pca', 'gmm', 'gmm_pca'],
        'accuracy':
            [mean_kmeans, mean_kmeans_pca, mean_gmm, mean_gmm_pca, max_kmeans, max_kmeans_pca, max_gmm, max_gmm_pca]
    }
    df = pd.DataFrame(data=d)

    sns_plot = sns.catplot(x='metric', y='accuracy', hue='method', data=df, kind='bar', legend=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    sns_plot.savefig('hw6/results/result-kmeans_gmm-compare-accuracy.png')
