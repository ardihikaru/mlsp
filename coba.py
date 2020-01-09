import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')

d = {
    'metric':
        ['mean', 'mean', 'mean', 'mean', 'max', 'max', 'max', 'max'],
    'method':
        ['kmeans', 'kmeans_pca', 'gmm', 'gmm_pca', 'kmeans', 'kmeans_pca', 'gmm', 'gmm_pca'],
    'accuracy':
        [0.914680, 0.300120, 0.118990, 0.667971, 0.329380, 0.189747, 0.660562, 0.882608]
}
df = pd.DataFrame(data=d)
sns.catplot(x='metric', y='accuracy', hue='method', data=df, kind='bar', legend=False)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
