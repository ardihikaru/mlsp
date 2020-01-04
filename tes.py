# import theano, csv
import csv
import numpy as np
import os
# from theano import tensor as T
from random import randint, shuffle


def load_data(filename):
    M = 4

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        dataset = list(reader)
        print(" >> dataset = ", dataset)
        # As the data-labels are arranged in sequential way
        # we make it randomly distributed via in-place shuffle
        shuffle(dataset)
        N = len(dataset)

        train = np.zeros((N, M))
        labels = np.zeros(N, dtype=np.int32)
        labels_map = dict()
        idx = 0

        for x in range(N):
            for y in range(M):
                train[x][y] = float(dataset[x][y])

            l = dataset[x][M]
            if l in labels_map:
                labels[x] = labels_map[l]
            else:
                labels[x] = idx
                labels_map[l] = idx
                idx = idx + 1

        return (train, labels, labels_map)

def main():
    project_dir = os.getcwd()
    dataset_path = project_dir + "/hw5/dataset/iris.data"
    print(" lokasi dataset = ", dataset_path)
    k = 5
    train, labels, labels_map = load_data(dataset_path)
    print("HASIL ..")
    print(" >> train = ", train)
    print(" >> labels = ", labels)
    print(" >> labels_map = ", labels_map)


main()
