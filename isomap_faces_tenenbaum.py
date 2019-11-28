"""
Replicate Joshua Tenenbaum's - the primary creator of the isometric feature mapping algorithm -  canonical, dimensionality reduction
research experiment for visual perception.
His original dataset from December 2000 consists of 698 samples of 4096-dimensional vectors.
These vectors are the coded brightness values of 64x64-pixel heads that have been rendered facing various directions and lighted from
many angles.
Can be accessed here: https://web.archive.org/web/20160913051505/http://isomap.stanford.edu/datasets.html
-Applying both PCA and Isomap to the 698 raw images to derive 2D principal components and a 2D embedding of the data's intrinsic
 geometric structure.
-Project both onto a 2D and 3D scatter plot, with a few superimposed face images on the associated samples.
"""
import pandas as pd
import scipy.io
import random, math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting ## Source: https://stackoverflow.com/questions/56222259/valueerror-unknown-projection-3d-once-again/56222305

from sklearn.decomposition import PCA
from sklearn import manifold

def Plot2D(T, title, x, y, num_to_plot=40):
    # This method picks a bunch of random samples (images in our case)
    # to plot onto the chart:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Component: {0}'.format(x))
    ax.set_ylabel('Component: {0}'.format(y))
    x_size = (max(T[:, x]) - min(T[:, x])) * 0.08
    y_size = (max(T[:, y]) - min(T[:, y])) * 0.08
    for i in range(num_to_plot):
        img_num = int(random.random() * num_images)
        x0, y0 = T[img_num, x] - x_size / 2., T[img_num, y] - y_size / 2.
        x1, y1 = T[img_num, x] + x_size / 2., T[img_num, y] + y_size / 2.
        # img = df.iloc[img_num, :].reshape(num_pixels, num_pixels)
        img = df.iloc[img_num, :].to_numpy().reshape(num_pixels, num_pixels)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    # It also plots the full scatter:
    ax.scatter(T[:, x], T[:, y], marker='.', alpha=0.7)


# A .MAT file is a .MATLAB file.
mat = scipy.io.loadmat('./hw4/Datasets/face_data.mat')
df = pd.DataFrame(mat['images']).T
num_images, num_pixels = df.shape
num_pixels = int(math.sqrt(num_pixels))

# Rotate the pictures, so we don't have to crane our necks:
for i in range(num_images):
    # df.loc[i, :] = df.loc[i, :].reshape(num_pixels, num_pixels).T.values.reshape(-1)
    # print(type(df.loc[i, :].to_numpy()))
    # df_new = df.loc[i, :]

    df.loc[i, :] = df.loc[i, :].to_numpy().reshape(num_pixels, num_pixels).T.reshape(-1)

#
# Implement PCA here. Reduce the dataframe df down
# to THREE components. Once you've done that, call Plot2D.
#
# The format is: Plot2D(T, title, x, y, num_to_plot=40):
# T is your transformed data, NDArray.
# title is your chart title
# x is the principal component you want displayed on the x-axis, Can be 0 or 1
# y is the principal component you want displayed on the y-axis, Can be 1 or 2
#
pca = PCA(n_components=3)
pca.fit(df)

T = pca.transform(df)

Plot2D(T, "PCA transformation", 1, 2, num_to_plot=140)

#
# Implement Isomap here. Reduce the dataframe df down
# to THREE components.
#
iso = manifold.Isomap(n_neighbors=8, n_components=3)
iso.fit(df)
manifold = iso.transform(df)

Plot2D(manifold, "ISO transformation", 1, 2, num_to_plot=40)

#
# draw your dataframes in 3D
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('0')
ax.set_ylabel('1')
ax.set_zlabel('2')

ax.scatter(manifold[:, 0], manifold[:, 1], manifold[:, 2], c='red')

plt.show()