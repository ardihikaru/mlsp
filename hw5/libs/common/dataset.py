from mlxtend.data import loadlocal_mnist
import os

class Dataset():
    def __init__(self, max_data=None, train_percent=0.8):
        self.__init_vars()
        if max_data is None:
            self.X, self.Y = self.__get_data()
        else:
            self.X, self.Y = self.__get_data(max_data)
        self.__setup_dataset(train_percent)

    def __init_vars(self):
        self.project_dir = os.getcwd()
        self.dataset_path = self.project_dir + "/hw5/dataset/"

    def __extract_mnist_dataset(self):
        X, Y = loadlocal_mnist(
            images_path=self.dataset_path + 'train-images.idx3-ubyte',
            labels_path=self.dataset_path + 'train-labels.idx1-ubyte')
        return X, Y

    def __get_data(self, limit=None):
        print("Reading in and transforming data...")
        X, Y = self.__extract_mnist_dataset()
        if limit is not None:
            X, Y = X[:limit], Y[:limit]
        return X, Y

    def __setup_dataset(self, train_percent):
        Ntrain_percent = train_percent  # 80% train, then 20% test
        Ntrain = int(Ntrain_percent * float(len(self.X)))

        self.Xtrain, self.Ytrain = self.X[:Ntrain], self.Y[:Ntrain]
        self.Xtest, self.Ytest = self.X[Ntrain:], self.Y[Ntrain:]

    def get_dataset(self):
        return self.Xtrain, self.Ytrain, self.Xtest, self.Ytest
