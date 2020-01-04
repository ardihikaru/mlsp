# MNIST classification using Support Vector algorithm with RBF kernel
# Author: Krzysztof Sopyla <krzysztofsopyla@gmail.com>
# https://ksopyla.com
# License: MIT
# Source: https://github.com/ksopyla/svm_mnist_digit_classification/blob/master/svm_mnist_classification.py

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
# fetch original mnist dataset
# from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score

# Disable warning: UndefinedMetricWarning: F-score is ill-defined
# and being set to 0.0 in labels with no predicted samples
import warnings
warnings.filterwarnings('always')

# import custom module
from hw5.libs.common.mnist_helpers import *

import matplotlib.pyplot as plt
from hw5.libs.algo.knn import KNN
from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple, save_to_csv
from datetime import datetime

if __name__ == '__main__':
    dataset = Dataset(train_data=80, test_data=20)
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    train_scores = []
    test_scores = []
    exec_times = []

    # Show example images (randomly selected)
    # show_some_digits(X_train, Y_train)

    ################ Classifier with good params ###########
    # Create a classifier: a support vector classifier

    # param_C = 5
    # param_gamma = 0.05
    # classifier = svm.SVC(C=param_C, gamma=param_gamma)
    classifier = svm.LinearSVC()

    # Create a classifier: a support vector classifier
    # kernel_svm = svm.SVC(gamma=.2)

    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    classifier.fit(X_train, Y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

    ########################################################
    # Now predict the value of the test
    expected = Y_test
    predicted = classifier.predict(X_test)
    # metrics.f1_score(Y_test, predicted, average='weighted', labels=np.unique(predicted))
    # metrics.f1_score(Y_test, predicted, labels=np.unique(predicted))

    # Show predicted results
    show_some_digits(X_test, predicted, title_text="Predicted {}")

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    plot_confusion_matrix(cm)

    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

# it creates mldata folder in your root project folder
# mnist = fetch_mldata('MNIST original', data_home='./')

# minist object contains: data, COL_NAMES, DESCR, target fields
# you can check it by running
# mnist.keys()

# data field is 70k x 784 array, each row represents pixels from 28x28=784 image
# images = mnist.data
# targets = mnist.target

# Let's have a look at the random 16 images,
# We have to reshape each data row, from flat array of 784 int to 28x28 2D array

# pick  random indexes from 0 to size of our dataset


# ---------------- classification begins -----------------
# scale data for [0,255] -> [0,1]
# sample smaller size for testing
# rand_idx = np.random.choice(images.shape[0],10000)
# X_data =images[rand_idx]/255.0
# Y      = targets[rand_idx]

# full dataset classification
# X_data = images / 255.0
# Y = targets

# split data to train and test
# from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)

