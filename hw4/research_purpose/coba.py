# import scipy.io
#
# matfile = "./hw4/datasetjg/UMist8.mat"
# mat = scipy.io.loadmat(matfile)
#
# print(type(mat))
# print(len(mat))
# print(mat)

# import PCA_centra
# import PCA_norm
# import SVD
# import ISOMAP
from builtins import print

from hw4.libs.new_isomap import *
from hw4.libs.metrics import _1NN

# DATADIR = '../dataset'
DATADIR = './hw4/datasetjg'

def ftrain(name):
    return path.join(DATADIR, '{}-train.txt'.format(name))

def ftest(name):
    return path.join(DATADIR, '{}-test.txt'.format(name))

for fname, KNN_K in {'sonar': 6, 'splice': 4}.items():

    print(" *** fname = ", fname)
    print(" *** KNN_K = ", KNN_K)

    traindataMat, traintagMat = loadData(ftrain(fname))
    testdataMat, testtagMat = loadData(ftest(fname))

    # print(traindataMat, traintagMat)
    # print(testdataMat, testtagMat)

    print(type(traindataMat))
    print(len(traindataMat))
    print(len(traindataMat[0]))

    # for dims in [10, 20, 30]:
    for dims in [2]:
        timestamp = time.time()

        print(traindataMat.shape)

        # dataMat_dimReduced = isomap(vstack([traindataMat, testdataMat]), dims, KNN_K)
        dataMat_dimReduced = isomap(traindataMat, dims, KNN_K)

        print(dataMat_dimReduced.shape)

        # print(" **** dataMat_dimReduced = ", dataMat_dimReduced)

        # accuracy = _1NN(dataMat_dimReduced[range(0, len(traindataMat)), :],
        #                 dataMat_dimReduced[range(len(traindataMat), len(dataMat_dimReduced)), :], mat(traintagMat),
        #                 mat(testtagMat))
        # print("(ISOMAP on %s(%d-NN))当维度为%d时,正确率为：" % (fname, KNN_K, dims), accuracy)
        print("time used：", time.time() - timestamp, 's')

        # print('ISOMAP, %s(%d-NN), k = %d, %.16f\n' % (fname, KNN_K, dims, accuracy))

    break
