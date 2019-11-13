"""
Mixed Images Separation performed via Independent Component Analysis.
The Fourth Order Blind Identification(FOBI) ICA is implemented here.
Source: https://github.com/ShashShukla/ICA
"""
# Import packages.
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as LA
from matplotlib.image import imread

# Import the images
im1 = np.asarray(imread("hw3/sample2/blend1.png"))
im2 = np.asarray(imread("hw3/sample2/blend2.png"))

# Generate linear signals out of these
im1 = np.reshape(im1, np.size(im1))
im2 = np.reshape(im2, np.size(im2))

print(im1.shape)
print(" disini ..")

# uint8 takes values from 0 to 255
im1 = im1 / 255.0
im1 = im1 - np.mean(im1)
im2 = im2 / 255.0
im2 = im2 - np.mean(im2)

# Output information about the image dimensions.
a = im1.shape
n = a[0]
print("Number of samples: ", n)
n = n * 1.0

time = np.arange(0, n, 1)

# x is our initial data matrix.
x = [im1, im2]

print("# Plot the signals from both sources to show correlations in the data.")
# Plot the signals from both sources to show correlations in the data.
plt.figure()
plt.plot(x[0], x[1], '*b')
plt.ylabel('Signal 2')
plt.xlabel('Signal 1')
plt.title("Original data")
plt.show()

print("# Calculate the covariance matrix of the initial data.")
# Calculate the covariance matrix of the initial data.
cov = np.cov(x)
# Calculate eigenvalues and eigenvectors of the covariance matrix.
d, E = LA.eigh(cov)
# Generate a diagonal matrix with the eigenvalues as diagonal elements.
D = np.diag(d)

print("# Perform whitening. xn is the whitened matrix.")
Di = LA.sqrtm(LA.inv(D))
# Perform whitening. xn is the whitened matrix.
xn = np.dot(Di, np.dot(np.transpose(E), x))

print("# Plot whitened data to show new structure of the data.")
# Plot whitened data to show new structure of the data.
plt.figure()
plt.plot(xn[0], xn[1], '*b')
plt.ylabel('Signal 2')
plt.xlabel('Signal 1')
plt.title("Whitened data")
plt.show()

print("# Perform FOBI.")
# Perform FOBI.
norm_xn = LA.norm(xn, axis=0)
# norm_xn = LA.norm(xn, axis=0)
norm = [norm_xn, norm_xn]

# Printing result
cov2 = np.cov(np.multiply(norm, xn))
d_n, Y = LA.eigh(cov2)
source = np.dot(np.transpose(Y), xn)

o_img_size = 800
np_array1 = np.array(source[0])
np_array2 = np.array(source[1])
out1 = np_array1.reshape((o_img_size, o_img_size, 3))
out2 = np_array2.reshape((o_img_size, o_img_size, 3))
# out1 = 10 * np.reshape(source[0], (img_size, img_size))
# out2 = 10 * np.reshape(source[1], (img_size, img_size))

# img_size = 32
# out1 = source[0][:, 0].reshape(img_size, img_size)

print("#Printing the result")
#Printing the result
# print(source[0].shape)
# print(out1.shape)
#
img_plot1 = plt.imshow(out1)
plt.show()
img_plot2 = plt.imshow(out2)
plt.show()

# cv2.imshow("Figure 1", out1)
# cv2.imshow("Figure 2", out2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()