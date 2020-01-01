
import numpy as np

class Dataset():
    def __init__(self):
        self.__load_img_dataset()
        self.__extract_img_dataset()
        self.__capture_indexes()
        self.__storing_digit_5s()

    def __load_img_dataset(self):
        self.img_size = 28
        # self.shapes = (self.img_size, self.img_size)
        self.npz_dataset = "./hw4/dataset/digits-labels.npz"

    def __extract_img_dataset(self):
        compressed = np.load(self.npz_dataset)
        self.d = compressed['d']  # capture extracted features
        self.l = compressed['l']  # captures the labels of corresponding d matrix
        self.max_digit = len(self.d)

    def __capture_indexes(self):
        # %% Capture only digit 5 and store each index-i into variables "idx_5s"
        self.idx_5s = []
        for i in range(0, self.max_digit):
            if int(self.l[i]) == 5:
                self.idx_5s.append(i)

    def __storing_digit_5s(self):
        # %% Store data of digit 5 into variable "fives"
        self.fives = np.zeros(((self.img_size ** 2), len(self.idx_5s)))
        for i in range(0, len(self.idx_5s)):
            self.fives[:, i] = self.d[:, self.idx_5s[i]]

    def get_5s(self):
        return self.fives

    def get_dataset(self):
        return self.d

