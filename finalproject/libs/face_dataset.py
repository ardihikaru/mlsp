import numpy as np
import glob
from matplotlib.image import imread

class FaceDataset():
    # def __init__(self, img_size=28, N=982, img_dataset_path=None, man_dataset=None):
    def __init__(self, img_size=112, N=19, img_dataset_path=None, man_dataset=None):
        self.img_size = img_size
        self.N = N # total number of image
        self.D = self.img_size**2 # dimension of the images dataset; D = (img_size x img_size)

        self.img = np.zeros((self.img_size, self.img_size))

        # self.source_img_path = "./hw3/four_dataset/"
        self.source_img_path = "./finalproject/datasets/1/"
        self.img_ext = ".jpg"

        if img_dataset_path is not None:
            self.source_img_path = img_dataset_path

        if man_dataset is not None:
            self.data = man_dataset # this is purposely used for Homework 4
        else:
            self.data = np.zeros((self.D, self.N))
            self.__load_default_imgs()

    def __load_default_imgs(self):
        i = 0
        img_data = self.source_img_path + "*" + self.img_ext
        # print(" **** img_data = ", img_data)
        for filename in glob.glob(img_data):
            self.img = np.asarray(imread(filename))
            self.data[:, i] = np.ravel(self.img)
            i = i + 1

    def get_img_features(self):
        return self.data


