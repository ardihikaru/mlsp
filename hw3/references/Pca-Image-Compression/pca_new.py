"""
@author: Muhammad Febrian Ardiansyah
"""

# from my_pca import Pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.image import imread
import numpy as np

X = []
for filename in glob.glob("four_dataset/*.jpg"):
    img = imread(filename)
    X.append(img.flatten('F'))

# Image is stored in "four_dataset"
X = np.array(X)
X = pd.DataFrame(X)

# Normalize data by subtracting mean and scaling
X_norm = normalize(X)

# Set pca to find principal components that explain 99%
# of the variation in the data
n_comp = 2
pca = PCA(n_components=n_comp)
# pca = Pca(X_norm)

# Run PCA on normalized image data
lower_dimension_data = pca.fit_transform(X_norm)

# Project lower dimension data onto original features
approximation = pca.inverse_transform(lower_dimension_data)

# Reshape approximation and X_norm to 5000x32x32 to display images
approximation = approximation.reshape(-1, 28, 28)
X_norm = X_norm.reshape(-1, 28, 28)

# Rotate pictures
for i in range(0,X_norm.shape[0]):
    X_norm[i, ] = X_norm[i, ].T
    approximation[i, ] = approximation[i, ].T

# Display the image
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle('PCA comparison with #Dim = %s' % n_comp)
ax1.imshow(X_norm[0, ], cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(approximation[0, ], cmap='gray')
ax2.set_title('Compressed Image')
plt.show()

