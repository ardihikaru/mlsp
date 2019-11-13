# source: https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array

from matplotlib.image import imread

img = imread('four0.jpg')
print(type(img))
print(img.shape)
# print(img)
print(len(img))
print(len(img[0]))

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=3)
pca.fit(img)

data_reduced = np.dot(img, pca.components_.T) # transform
data_original = np.dot(data_reduced, pca.components_) # inverse_transform

# Reshape approximation and X_norm to 5000x32x32 to display images
# approximation = approximation.reshape(-1, 32, 32)
# X_norm = X_norm.reshape(-1, 32, 32)

# Rotate pictures
for i in range(0, data_original.shape[0]):
    data_original[i,] = data_original[i,].T
    data_reduced[i,] = data_reduced[i, ].T

# Display images
fig4, axarr = plt.subplots(1, 2, figsize=(8, 8))
axarr[0, 0].imshow(data_original[0, ], cmap='gray')
axarr[0, 0].set_title('Original Image')
axarr[0, 0].axis('off')
axarr[0, 1].imshow(data_reduced[0,], cmap='gray')
axarr[0, 1].set_title('99% Variation')
axarr[0, 1].axis('off')
# axarr[1, 0].imshow(X_norm[1,], cmap='gray')
# axarr[1, 0].set_title('Original Image')
# axarr[1, 0].axis('off')
# axarr[1, 1].imshow(approximation[1,], cmap='gray')
# axarr[1, 1].set_title('99% Variation')
# axarr[1, 1].axis('off')
# axarr[2, 0].imshow(X_norm[2,], cmap='gray')
# axarr[2, 0].set_title('Original Image')
# axarr[2, 0].axis('off')
# axarr[2, 1].imshow(approximation[2,], cmap='gray')
# axarr[2, 1].set_title('99% variation')
# axarr[2, 1].axis('off')
plt.show()
