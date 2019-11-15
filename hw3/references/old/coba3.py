"""
Created on Sun Apr 15 00:39:35 2018
@author: Hrid
Source: https://github.com/hridkamolbiswas/Principal-Component-Analysis-PCA-on-image-dataset/blob/master/pca.py
"""

import numpy as np
from numpy import linalg as LA
import os, os.path
# np.set_printoptions(threshold=np.nan)
# import cv2
from matplotlib.image import imread
from PIL import Image
import glob
# import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

data = np.empty((0, 2048))  # 2048 is the size of the feature vector/number of pixels after  resizing the image

arr = []
for filename in glob.glob("four_dataset/*.jpg"):
    # im = cv2.imread(filename, 0)
    img = imread(filename)
    print(type(img))
    print(img.shape)

    X = pd.DataFrame(img)
    print(" Printing X..")
    print(type(X))
    print(X.shape)
    print(X[0])

    # Normalize data by subtracting mean and scaling
    X_norm = normalize(X)
    print(" Printing [X_norm] ..")
    print(X_norm.shape)
    print(X_norm[0])
    print(" ---- ")

    # pca = PCA(n_components=2)
    # # Run PCA on normalized image data
    # lower_dimension_data = pca.fit_transform(X_norm)
    # # Lower dimension data is 5000x353 instead of 5000x1024
    # print(" Printing [lower_dimension_data] ..")
    # print(lower_dimension_data.shape)
    # print(" ---- ")
    #
    # # Run PCA on normalized image data
    # lower_dimension_data = pca.fit_transform(X_norm)
    #
    # # Project lower dimension data onto original features
    # approximation = pca.inverse_transform(lower_dimension_data)
    #
    # # Reshape approximation and X_norm to 5000x32x32 to display images
    # # approximation = approximation.reshape(-1, 32, 32)
    # # X_norm = X_norm.reshape(-1, 32, 32)
    #
    # # Rotate pictures
    # for i in range(0, X_norm.shape[0]):
    #     X_norm[i,] = X_norm[i,].T
    #     approximation[i,] = approximation[i, ].T
    #
    # # Display images
    # fig4, axarr = plt.subplots(3, 2, figsize=(8, 8))
    # axarr[0, 0].imshow(X_norm[0, ], cmap='gray')
    # axarr[0, 0].set_title('Original Image')
    # axarr[0, 0].axis('off')
    # axarr[0, 1].imshow(approximation[0,], cmap='gray')
    # axarr[0, 1].set_title('99% Variation')
    # axarr[0, 1].axis('off')
    # # axarr[1, 0].imshow(X_norm[1,], cmap='gray')
    # # axarr[1, 0].set_title('Original Image')
    # # axarr[1, 0].axis('off')
    # # axarr[1, 1].imshow(approximation[1,], cmap='gray')
    # # axarr[1, 1].set_title('99% Variation')
    # # axarr[1, 1].axis('off')
    # # axarr[2, 0].imshow(X_norm[2,], cmap='gray')
    # # axarr[2, 0].set_title('Original Image')
    # # axarr[2, 0].axis('off')
    # # axarr[2, 1].imshow(approximation[2,], cmap='gray')
    # # axarr[2, 1].set_title('99% variation')
    # # axarr[2, 1].axis('off')
    # plt.show()

    break


    # # print('size:',im.shape)
    # resized = img.reshape(-1,32,32) # ERROR: ValueError: cannot reshape array of size 784 into shape (32,32)
    # # resized = cv2.reshape(img, (32, 64))
    # im_ravel = resized.ravel()
    # arr = np.append(data, [im_ravel], axis=0)
    # data = arr

# final_data = arr
# mu = np.mean(final_data, axis=0)
#
# plt.figure(1)
#
# k = 1677
# for i in range(0, 4):
#     img1 = final_data[k, :]
#     ir = np.reshape(img1, (32, 64))
#     ir = np.uint8(ir)
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(ir, cmap='gray')
#     k = k + 1
#     print('k=== ', k)
# plt.suptitle('sample image from training dataset')
# plt.show()
#
# data = final_data - mu
# covariance = np.cov(data.T)
# values, vector = LA.eig(covariance)
#
# pov = np.cumsum(np.divide(values, sum(values)))
# plt.figure
# plt.plot(pov)
# plt.title('percentage of variance explained')
#
# vsort = vector[:, 0:301]
# scores = np.dot(data, vsort)
# projection = np.dot(scores, vsort.T) + mu
#
# % matplotlib
# qt
# plt.figure(2)
# k = 1677
# for i in range(0, 4):
#     img1_train = projection[k, :]
#     ir_train = np.reshape(img1_train, (32, 64))
#     ir = np.uint8(ir_train)
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(ir_train, cmap='gray')
#     k = k + 1
#     print('k=== ', k)
# plt.suptitle('Image construction using PCA')
# plt.show()