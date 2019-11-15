import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as LA

from hw3.libs.common.blend_dataset import BlendImgDataset

class ICA(BlendImgDataset):
    def __init__(self, o_img_size):
        super().__init__(o_img_size)

    """
    Step for ICA Algorithm:
    1. Calculate the covariance matrix of the initial data
    2. Perform whitening. xn is the whitened matrix.
    3. Plot whitened data to show new structure of the data. (opt_
    4. Perform FOBI.
    5. Printing the result
    """
    def run(self):
        print(" ## Start running ICA Algorithm")
        E, D = self.calc_covariance_matrix()
        xn = self.whitening(E, D)
        # self.plot_whitened_structure(xn)
        norm = self.exec_fobi(xn)
        self.plot_result(norm, xn)

    def calc_covariance_matrix(self):
        # Calculate the covariance matrix of the initial data.
        cov = np.cov(self.img_data)
        # Calculate eigenvalues and eigenvectors of the covariance matrix.
        d, E = LA.eigh(cov)
        # Generate a diagonal matrix with the eigenvalues as diagonal elements.
        D = np.diag(d)
        return E, D

    def whitening(self, E, D):
        print("# Perform whitening. xn is the whitened matrix.")
        Di = LA.sqrtm(LA.inv(D))
        # Perform whitening. xn is the whitened matrix.
        xn = np.dot(Di, np.dot(np.transpose(E), self.img_data))
        # print(" ************* SHAPE xn", xn.shape)

        # Try to plot only the first 4 images to test out the result
        selected_xn = xn[:4, :]
        # print(" ************* SHAPE selected_xn", selected_xn.shape)

        # return xn
        return selected_xn

    def plot_whitened_structure(self, xn):
        print("# Sample Plot (of two first blended images) whitened data to show new structure of the data.")
        # Plot whitened data to show new structure of the data.
        plt.figure()
        plt.plot(xn[0], xn[1], '*b')
        plt.ylabel('Signal 2')
        plt.xlabel('Signal 1')
        plt.title("Whitened data")
        plt.show()

    def exec_fobi(self, xn):
        print("# Perform FOBI.")
        # Perform FOBI.
        norm_xn = LA.norm(xn, axis=0)
        # norm_xn = LA.norm(xn, axis=0)
        # norm = [norm_xn, norm_xn]

        # print(" ** self.total_img = ", self.total_img)
        # norm = [norm_xn, norm_xn]
        # norm = [norm_xn, norm_xn, norm_xn, norm_xn]
        norm = []
        for i in range(0, self.expected_type):
            norm.append(norm_xn)

        # print(norm)

        print(" FOBI FINISHED ..")

        return norm

    def plot_result(self, norm, xn):
        # print(" *** CHECK ..")
        # print(len(norm))
        # print(xn)
        # print(xn.shape)
        # print(xn[:4, :].shape)
        # print(" ********")

        cov2 = np.cov(np.multiply(norm, xn))
        d_n, Y = LA.eigh(cov2)
        source = np.dot(np.transpose(Y), xn)

        # Try to plot only the first 4 images to test out the result
        # selected_xn = xn[:4, :]
        # cov2 = np.cov(np.multiply(norm, selected_xn))
        # d_n, Y = LA.eigh(cov2)
        # source = np.dot(np.transpose(Y), selected_xn)

        # print(" ** TOTAL output type = ", len(source))
        # print(source[0])
        # print(len(source[0]))
        # print(len(source))
        # print(source[0])

        # np_arr = np.array(source[0])
        # ori_data = np_arr.reshape(self.o_img_size, self.o_img_size)
        # # ori_data = source[0].reshape(self.o_img_size, self.o_img_size)
        # print(len(ori_data))
        # print(len(ori_data[0]))
        # print(ori_data[0][0])
        # # imgplot = plt.imshow(ori_data)
        # # plt.show()
        #
        # # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # f, (ax1, ax2) = plt.subplots(1, 2)
        # f.suptitle('Factorized Blended Images')
        # ax1.imshow(ori_data, cmap='gray', interpolation='none')
        # ax1.set_title('Factorized Result 1')
        # ax2.imshow(ori_data, cmap='gray', interpolation='none')
        # ax2.set_title('Factorized Result 2')
        # plt.show()

        output_img = []
        for i in range(0, self.expected_type):
            output_img.append(source[i].reshape(self.o_img_size, self.o_img_size))
        # np_arr = np.array(source[0])
        # ori_data = source[0].reshape(self.o_img_size, self.o_img_size)
        # lain_data = source[1].reshape(self.o_img_size, self.o_img_size)
        # lain_data2 = source[2].reshape(self.o_img_size, self.o_img_size)
        # ori_data = np_arr.reshape(1, self.o_img_size, self.o_img_size)
        # print(ori_data)
        # print("\n ***** ")
        # print(len(ori_data))
        # print(len(ori_data[0]))

        # f, (ax1, ax2) = plt.subplots(1, 2)
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        f.suptitle('Factorized Blended Images')
        ax1.imshow(output_img[0], cmap='gray', interpolation='none')
        ax1.set_title('Result 1')
        ax2.imshow(output_img[1], cmap='gray', interpolation='none')
        ax2.set_title('Result 2')
        ax3.imshow(output_img[2], cmap='gray', interpolation='none')
        ax3.set_title('Result 3')
        ax4.imshow(output_img[3], cmap='gray', interpolation='none')
        ax4.set_title('Result 4')
        plt.show()

        # ori_data = source[0][:, 0].reshape(self.o_img_size, self.o_img_size)
        # print(ori_data)
        # print(len(ori_data))

        # img = np.array(source[0]).flatten().reshape(1, 784)
        # print(img)
        # print(len(img))

        # imgplot = plt.imshow(ori_data)
        # plt.show()

        # img_output = []
        # for i in range(0, self.expected_type):
        #     np_arr = np.array(source[i])
        #     img_output.append(np_arr.reshape((self.o_img_size, self.o_img_size, 3)))
        #     # img_output.append(source[i].reshape(self.o_img_size, self.o_img_size))
        #
        #     # ori_data = source[i].reshape(self.o_img_size, self.o_img_size)
        #
        # fig4, axarr = plt.subplots(2, 2, figsize=(9, 9))
        # # axarr[0, 0].imshow(img_output[0], cmap='gray', interpolation='none')
        # axarr[0, 0].imshow(features, cmap='gray', interpolation='none')
        # axarr[0, 0].set_title('Original Image')
        # axarr[0, 0].axis('off')
        #
        # # axarr[0, 1].imshow(img_output[1], cmap='gray', interpolation='none')
        # axarr[0, 1].imshow(features, cmap='gray', interpolation='none')
        # axarr[0, 1].set_title('Compressed hai')
        # axarr[0, 1].axis('off')
        #
        # # axarr[1, 0].imshow(img_output[2], cmap='gray', interpolation='none')
        # # axarr[1, 0].imshow(img_output[0], cmap='gray', interpolation='none')
        # axarr[1, 0].imshow(features, cmap='gray', interpolation='none')
        # axarr[1, 0].set_title('Original Image')
        # axarr[1, 0].axis('off')
        #
        # # axarr[1, 1].imshow(img_output[3], cmap='gray', interpolation='none')
        # # axarr[1, 1].imshow(img_output[1], cmap='gray', interpolation='none')
        # axarr[1, 1].imshow(features, cmap='gray', interpolation='none')
        # axarr[1, 1].set_title('Compressed hai')
        # axarr[1, 1].axis('off')
        # plt.show()
