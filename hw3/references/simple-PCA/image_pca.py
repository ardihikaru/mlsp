import numpy as np 
from PIL import Image
import settings as globalv
import pca_utils as pca
from matplotlib import pyplot as plt
import glob
from matplotlib.image import imread
import warnings # Disable warning: https://stackoverflow.com/questions/41001533/how-to-ignore-python-warnings-for-complex-numbers
warnings.filterwarnings('ignore')
# from sklearn.preprocessing import normalize

globalv.init(False)

# globalv.N = 14
globalv.N = 982
N = globalv.N
# imgsize = 50
imgsize = 28
globalv.D = imgsize**2
D = globalv.D

img = np.zeros((imgsize,imgsize))
data = np.zeros((D, N))

plt.figure(1)

i = 0
for filename in glob.glob("four_dataset/*.jpg"):
    img = np.asarray(imread(filename))
    data[:, i] = np.ravel(img)
    i = i + 1

# for i in range(N):
# 	img = np.asarray(Image.open('faces/' + str(i+1) +'.jpg').convert('L'))
# 	data[:,i] = np.ravel(img)

print(" *** End reading image ..")

# The loaded images form a data vector of dimensions DxN, D=76^2, N = 6
# that is, 6 patterns of 76^2-dimensional data
# Because there are less patterns than dimensions, the maximum number of uncorrelated
# components we can keep is 6-1 = 5, because there are no more directions of data variability

# Do PCA:
centered_data, mean_vector = pca.centerData(data)
corr = pca.correlationMatrix(centered_data)
eigvals, eigvecs = pca.eigenDecomposition(corr)
# r = pca.readUserNumComponents('\nHow many components (0 <= int <= %d)?\n' %D)
n_comp = 4
# proj_matrix = pca.computeProjectionMatrix(eigvals, eigvecs, r)
proj_matrix = pca.computeProjectionMatrix(eigvals, eigvecs, n_comp)
princ_comps = pca.computePrincipalComponents(centered_data, proj_matrix)

reconstructed_data = pca.reconstructData(princ_comps, proj_matrix, mean_vector)

error = pca.computeRecError(data, reconstructed_data)

# Printing the graph
ori_data = data[:, 0].reshape(imgsize, imgsize)
rec_data = reconstructed_data[:, 0].reshape(imgsize, imgsize)

f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle('PCA comparison with #Dim = %s' % n_comp)
ax1.imshow(ori_data, cmap='gray', interpolation='none')
ax1.set_title('Original Image')
ax2.imshow(rec_data, cmap='gray', interpolation='none')
ax2.set_title('Compressed Image')
plt.show()

# n_max = 1
# for i in range(n_max):
#     rec_data = reconstructed_data[:, i].reshape(imgsize, imgsize)
#     plt.subplot(2, n_max, n_max+i+1)
#     plt.axis('off')
#     plt.imshow(rec_data, cmap='gray', interpolation='none')
#
# plt.show()