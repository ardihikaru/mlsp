# Source: As of version 0.20, sklearn deprecates fetch_mldata function and adds fetch_openml instead.

# from sklearn.datasets import fetch_mldata

# numbers = fetch_mldata('MNIST original')
# print(numbers.data.shape)
# print(len(numbers.data[1250]))

from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784')

print(mnist.data.shape)
print(len(mnist.data[1250]))

data = mnist.data

# Source: https://stackoverflow.com/questions/37228371/visualize-mnist-dataset-using-opencv-or-matplotlib-pyplot
# The first column is the label
label = "Iki label"
# label = data[0]
# print(' ** label = ', label)

# The rest of columns are pixels
pixels = data[0]
# pixels = data[1:]
print(' ** pixels = ', pixels)

# Make those columns into a array of 8-bits pixels
# This array will be of 1D with length 784
# The pixel intensity values are integers from 0 to 255
pixels = np.array(pixels, dtype='uint8')
print(' ** pixels = ', pixels)

# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = pixels.reshape((28, 28))
print(' ** pixels = ', pixels)

# Plot
plt.title('Label is {label}'.format(label=label))
plt.imshow(pixels, cmap='gray')
plt.show()

# i = 15
# plt.imshow(np.reshape(mnist[:, i], (28, 28), 'F'))
# plt.imshow(np.reshape(mnist[:, i], (28, 28, 1), 'F'))
# plt.show()

print("DONE ..")

# Source: https://github.com/ntmeyer12/MATH273C/blob/master/manifoldlearning.ipynb
fig, numplt = plt.subplots(10, 10, subplot_kw=dict(xticks=[], yticks=[]))
for i, axis in enumerate(numplt.flat):
    axis.imshow(data[700*i].reshape(28, 28), cmap='gray_r')
    # axis.show()
    # axis.imshow(numbers.data[700*i].reshape(28, 28), cmap='gray_r')
