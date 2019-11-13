# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 18:02:22 2018
@author: dillon
Source: https://github.com/dilloncamp/pca
"""

from sklearn.decomposition import PCA
# from .
from sklearn.preprocessing import normalize
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.image as image
import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.image import imread
import numpy as np

# baca image dari matlab file: .mat
# images = loadmat('ex7faces.mat',variable_names='IMAGES',appendmat=True).get('IMAGES')
# imgplot = plt.imshow(images[0])
# imgplot = plt.imshow(images[:, :, 0])
# plt.show()


X = []
for filename in glob.glob("four_dataset/*.jpg"):
    # im = cv2.imread(filename, 0)
    img = imread(filename)
    X.append(img.flatten('F'))

#Image is stored in MATLAB dataset
# X = sio.loadmat('ex7faces.mat')
X = np.array(X)
# print(X.shape)
# print(np.frombuffer(X).shape)
# X = pd.DataFrame(X['X'])
X = pd.DataFrame(X)
# print(" Printing X..")
# print(type(X))
# print(X.shape)
# print(X[0])
# print(" ---- ")

#Normalize data by subtracting mean and scaling
X_norm = normalize(X)
# print(" Printing [X_norm] ..")
# print(X_norm.shape)
# print(X_norm[0])
# print(" ---- ")

#Set pca to find principal components that explain 99%
#of the variation in the data
# pca = PCA(.99)
n_comp = 2
pca = PCA(n_components=n_comp)
#Run PCA on normalized image data
lower_dimension_data = pca.fit_transform(X_norm)
#Lower dimension data is 5000x353 instead of 5000x1024
# print(" Printing [lower_dimension_data] ..")
# print(lower_dimension_data.shape)
# print(" ---- ")

#Project lower dimension data onto original features
approximation = pca.inverse_transform(lower_dimension_data)
#Approximation is 5000x1024
# print(" Printing [approximation] ..")
# print(approximation.shape)
# print(" ---- ")

#Reshape approximation and X_norm to 5000x32x32 to display images
# approximation = approximation.reshape(-1,32,32)
approximation = approximation.reshape(-1, 28, 28)
# X_norm = X_norm.reshape(-1,32,32)
X_norm = X_norm.reshape(-1, 28, 28)
# print(" Printing [X_norm] ..")
# print(X_norm.shape)
# print(" ---- ")

#Rotate pictures
for i in range(0,X_norm.shape[0]):
    X_norm[i, ] = X_norm[i, ].T
    approximation[i, ] = approximation[i, ].T


# display images
# plt.subplots(nrows=1, ncols=2, figsize=(20, 20))


#Display images
# _, axarr = plt.subplots(3,2,figsize=(8,8))
# _, axarr = plt.subplots(1, 2, figsize=(20, 20))
_, axarr = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
axarr[0, 0].imshow(X_norm[0, ], cmap='gray')
axarr[0, 0].set_title('Original Image')
axarr[0, 0].axis('off')
axarr[0, 1].imshow(approximation[0, ], cmap='gray')
axarr[0, 1].set_title('# Components: %s' % n_comp)
axarr[0, 1].axis('off')
plt.show()

# #Display images
# fig4, axarr = plt.subplots(3,2,figsize=(8,8))
# axarr[0,0].imshow(X_norm[0,],cmap='gray')
# axarr[0,0].set_title('Original Image')
# axarr[0,0].axis('off')
# axarr[0,1].imshow(approximation[0,],cmap='gray')
# axarr[0,1].set_title('99% Variation')
# axarr[0,1].axis('off')
# axarr[1,0].imshow(X_norm[1,],cmap='gray')
# axarr[1,0].set_title('Original Image')
# axarr[1,0].axis('off')
# axarr[1,1].imshow(approximation[1,],cmap='gray')
# axarr[1,1].set_title('99% Variation')
# axarr[1,1].axis('off')
# axarr[2,0].imshow(X_norm[2,],cmap='gray')
# axarr[2,0].set_title('Original Image')
# axarr[2,0].axis('off')
# axarr[2,1].imshow(approximation[2,],cmap='gray')
# axarr[2,1].set_title('99% variation')
# axarr[2,1].axis('off')
# plt.show()
