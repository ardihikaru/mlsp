import numpy as np
import glob
from matplotlib.image import imread
import matplotlib.pyplot as plt
import pandas as pd

class BlendImgDataset():
    def __init__(self, o_img_size, df_dataset=False, shape=None, N=None, p=None, all=False):
        self.total_img = 0
        self.o_img_size = o_img_size
        self.img_data = [] # initial data matrix

        self.shape = shape  # origianl image size
        self.N = N  # image sample size
        self.p = p  # = 784

        # self.source_img_path = "./hw3/blend_20/"
        # self.img_ext = ".jpg"
        # self.expected_type = 4

        # self.source_img_path = "./hw3/blend_4/"
        # self.img_ext = ".jpg"
        # self.expected_type = 4

        # self.source_img_path = "./hw3/blend_10/"
        # self.img_ext = ".jpg"
        # self.expected_type = 4

        # self.source_img_path = "./hw3/mixture_dataset(0147)/"
        # self.img_ext = ".jpg"
        # self.expected_type = 4

        # self.source_img_path = "./hw3/blend_sample/"
        # self.img_ext = ".png"
        # self.expected_type = 2

        if all:
            self.source_img_path = "./hw3/mixture_dataset(0147)/"
            self.img_ext = ".jpg"
            self.expected_type = 4
        else:
            self.source_img_path = "./hw3/blend_small/"
            self.img_ext = ".jpg"
            self.expected_type = 4

        if df_dataset:
            self.__load_default_imgs_df(p)
        else:
            self.__load_default_imgs()
            # self.check_signal_cor()

    def __load_default_imgs_df(self, p):
        X = []
        indices = []
        img_data = self.source_img_path + "*" + self.img_ext
        for filename in glob.glob(img_data):
            index = filename[3:-4]
            self.total_img += 1
            print(" reading dataset, index = ", index)
            img = plt.imread(filename)   # notice that the input file order is random!!!!!!!
            X.append(img.ravel())
            indices.append(index)
            self.img_data = pd.DataFrame(X, columns=["V" + str(p) for p in range(p)], index=indices)

    def __load_default_imgs(self):
        # i = 0
        img_data = self.source_img_path + "*" + self.img_ext
        for filename in glob.glob(img_data):
            self.total_img += 1
            img = np.asarray(imread(filename))

            # Generate linear signals out of these
            im = np.reshape(img, np.size(img))
            # uint8 takes values from 0 to 255
            im = im / 255.0
            im = im - np.mean(im)

            # collect it as the dataset
            self.img_data.append(im)

    def get_initial_data_matrix(self):
        return self.img_data

    def check_signal_cor(self):
        print("# Sample Plot (of two first blended images) the signals from both sources to show correlations in the data.")
        # Plot the signals from both sources to show correlations in the data.
        plt.figure()
        plt.plot(self.img_data[0], self.img_data[1], '*b')
        plt.ylabel('Signal 2')
        plt.xlabel('Signal 1')
        plt.title("Original data")
        plt.show()
