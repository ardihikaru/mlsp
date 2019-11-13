import numpy as np
from matplotlib import pyplot as plt
import glob
from matplotlib.image import imread

class ImgDataset():
    def __init__(self, img_size=28, N=982):
        self.img_size = img_size
        self.N = N # total number of image
        self.D = self.img_size**2 # dimension of the images dataset; D = (img_size x img_size)

        self.img = np.zeros((self.img_size, self.img_size))
        self.data = np.zeros((self.D, self.N))

        self.source_img_path = "./hw3/four_dataset/"
        self.img_ext = ".jpg"

        self.__load_default_imgs()

    def __load_default_imgs(self):
        i = 0
        img_data = self.source_img_path + "*" + self.img_ext
        for filename in glob.glob(img_data):
            self.img = np.asarray(imread(filename))
            self.data[:, i] = np.ravel(self.img)
            i = i + 1



