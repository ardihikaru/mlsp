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
    # dataset = Dataset(train_data=80, test_data=20)
    # dataset = Dataset(train_data=800, test_data=200)
    # dataset = Dataset(train_data=2000, test_data=200)
    # dataset = Dataset(train_data=4000, test_data=400)
    # dataset = Dataset(train_data=10000, test_data=1000)
    dataset = Dataset()
    # dataset = Dataset()
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    train_scores = []
    test_scores = []
    exec_times = []

    ################ Classifier with good params ###########
    # Create a classifier: a support vector classifier

    classifier = svm.LinearSVC()

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

    # Show predicted results
    show_some_digits(X_test, predicted, title_text="Predicted {}")

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    plot_confusion_matrix(cm)

    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))

    total_exec_time = end_time - start_time
    print('total_exec_time {}'.format(str(total_exec_time)))
