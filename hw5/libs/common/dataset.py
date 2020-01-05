from mlxtend.data import loadlocal_mnist
import os

class Dataset():
    def __init__(self, train_data=None, test_data=None, is_lib=False):
        self.__init_vars()
        self.__setup_dataset(train_data, test_data)

    def __init_vars(self):
        self.project_dir = os.getcwd()
        self.dataset_path = self.project_dir + "/hw5/dataset/"

    def __setup_dataset_mnist_lib(self):
        import numpy as np
        from keras.datasets import mnist
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()
        self.X_train = np.reshape(self.X_train, (60000, 784))
        self.X_test = np.reshape(self.X_test, (10000, 784))

        # Standardize dataset
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train = self.X_train / 255.
        self.X_test = self.X_test / 255.

    # Max train dataset = 60k
    def __extract_mnist_train(self, limit):
        # fix bug: max limit = 60k
        X, Y = loadlocal_mnist(
            images_path=self.dataset_path + 'train-images.idx3-ubyte',
            labels_path=self.dataset_path + 'train-labels.idx1-ubyte')
        if limit is not None:
            if limit > 60000:
                limit = 60000
            X, Y = X[:limit], Y[:limit]
        return X, Y

    # Max test dataset = 10k
    def __extract_mnist_test(self, limit):
        # fix bug: max limit = 10k
        X, Y = loadlocal_mnist(
            images_path=self.dataset_path + 't10k-images.idx3-ubyte',
            labels_path=self.dataset_path + 't10k-labels.idx1-ubyte')
        if limit is not None:
            if limit > 60000:
                limit = 60000
            X, Y = X[:limit], Y[:limit]
        return X, Y

    def __setup_dataset(self, limit_train, limit_test):
        self.X_train, self.Y_train = self.__extract_mnist_train(limit_train)
        self.X_test, self.Y_test = self.__extract_mnist_test(limit_test)

        # Standardize dataset
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train = self.X_train / 255.
        self.X_test = self.X_test / 255.

    def get_dataset(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test
