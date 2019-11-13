# import normalization
# from . import normalization
from sklearn.preprocessing import normalize
from normalization import Normalization
import scipy as sp
from scipy import linalg
import csv
from matplotlib.image import imread
import glob

from sklearn.decomposition import PCA
# from .
from sklearn.preprocessing import normalize
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.image as image
import pandas as pd
import matplotlib.pyplot as plt

class Pca:
    
    def __init__(self, data):
        # self._normalization = normalization.Normalization(data)

        # self._normalization = normalize(data)
        self._normalization = Normalization(data)

        normalized_data = self._normalization.normalized_dataset()
        data_matrix = sp.matrix(normalized_data)
        m = data_matrix.shape[0]
        covariance_matrix = data_matrix.transpose()*data_matrix
        covariance_matrix /= m
        eig_decomp = linalg.eigh(covariance_matrix)  
        self._n = len(eig_decomp[0])
        self._pcas = sp.zeros((self._n, self._n))  
        for i in range(self._n):
            self._pcas[i, :] = eig_decomp[1][:, self._n - i - 1]

        self._eig_vals = list(eig_decomp[0])
        self._eig_vals.reverse()

    @property
    def pcas(self):
        return self._pcas

    @property
    def eig_vals(self):
        return self._eig_vals

    @property
    def n(self):
        return self._n

    def project(self, vector, k):
        v = self._normalization.normalize_x(vector)

        # project it
        v = sp.array(v)
        dot_product = lambda pca: sum(pca[j]*v[j] for j in range(len(v)))
        return [dot_product(self.pcas[i]) for i in range(k)]

    def deproject(self, vector):
        v = list(vector)
        result = sp.zeros(self._n)
        for i in range(len(v)):
            result += self._pcas[i]*v[i]

        result = self._normalization.denormalize_x(list(result))
        return list(result)


def main():
    rows = []
    # delimiter = ','
    # with open('four_dataset/housing.csv', 'r') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=delimiter)
    #     for row in csv_reader:
    #         rows.append([float(r) for r in row])
    # data = sp.array(rows)

    # Image is stored in MATLAB dataset
    X = sio.loadmat('ex7faces.mat')
    # print(X.shape)
    X = pd.DataFrame(X['X'])
    # Normalize data by subtracting mean and scaling
    X_norm = normalize(X)

    pca = Pca(X_norm)
    K = 10
    before_compression = X_norm[5, :]
    compressed = pca.project(before_compression, K)

    after_decompression = pca.deproject(compressed)

    print("Before compression: \n" + str(before_compression))
    print("Compressed: \n" + str(compressed))
    print("After decompression: \n" + str(after_decompression))

    print("Difference: \n" + str(list(after_decompression - before_compression)))

    approximation = after_decompression.reshape(-1, 32, 32)
    X_norm = before_compression.reshape(-1, 32, 32)

    # Display images
    fig4, axarr = plt.subplots(3, 2, figsize=(8, 8))
    axarr[0, 0].imshow(X_norm[0,], cmap='gray')
    axarr[0, 0].set_title('Original Image')
    axarr[0, 0].axis('off')
    axarr[0, 1].imshow(approximation[0,], cmap='gray')
    axarr[0, 1].set_title('99% Variation')
    axarr[0, 1].axis('off')
    axarr[1, 0].imshow(X_norm[1,], cmap='gray')
    axarr[1, 0].set_title('Original Image')
    axarr[1, 0].axis('off')
    axarr[1, 1].imshow(approximation[1,], cmap='gray')
    axarr[1, 1].set_title('99% Variation')
    axarr[1, 1].axis('off')
    axarr[2, 0].imshow(X_norm[2,], cmap='gray')
    axarr[2, 0].set_title('Original Image')
    axarr[2, 0].axis('off')
    axarr[2, 1].imshow(approximation[2,], cmap='gray')
    axarr[2, 1].set_title('99% variation')
    axarr[2, 1].axis('off')
    plt.show(

    # # Set pca to find principal components that explain 99%
    # # of the variation in the data
    # # pca = PCA(.99)
    # # pca = PCA(n_components=230)
    # pca = PCA(n_components=230)
    # # Run PCA on normalized image data
    # lower_dimension_data = pca.fit_transform(X_norm)
    # # Lower dimension data is 5000x353 instead of 5000x1024
    #
    # # Project lower dimension data onto original features
    # approximation = pca.inverse_transform(lower_dimension_data)
    # # Approximation is 5000x1024
    # # Reshape approximation and X_norm to 5000x32x32 to display images
    # approximation = approximation.reshape(-1, 32, 32)
    # X_norm = X_norm.reshape(-1, 32, 32)
    #
    # # Rotate pictures
    # for i in range(0, X_norm.shape[0]):
    #     X_norm[i,] = X_norm[i,].T
    #     approximation[i,] = approximation[i,].T
    #
    # # Display images
    # fig4, axarr = plt.subplots(3, 2, figsize=(8, 8))
    # axarr[0, 0].imshow(X_norm[0,], cmap='gray')
    # axarr[0, 0].set_title('Original Image')
    # axarr[0, 0].axis('off')
    # axarr[0, 1].imshow(approximation[0,], cmap='gray')
    # axarr[0, 1].set_title('99% Variation')
    # axarr[0, 1].axis('off')
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
    # plt.show()

    #################################################33
    #################################################33

    # for filename in glob.glob("four_dataset/*.jpg"):
    #     img = imread(filename)
    #     fl_img = [float(r) for r in img]
    #     print(fl_img)
    #     rows.append(img)
    #     rows.append(img)
    # data = sp.array(rows)
    #
    # pca = Pca(data)
    # K = 10
    # before_compression = data[5, :]
    # compressed = pca.project(before_compression, K)
    #
    # after_decompression = pca.deproject(compressed)
    #
    # print("Before compression: \n" + str(before_compression))
    # print("Compressed: \n" + str(compressed))
    # print("After decompression: \n" + str(after_decompression))
    #
    # print("Difference: \n" + str(list(after_decompression - before_compression)))
    #
    # # print(data)

if __name__ == '__main__':
    main()
