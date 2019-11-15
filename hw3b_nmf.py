import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as LA
from matplotlib.image import imread

# 1. Import custom libraries
from hw3.libs.algorithms.nmf import NMF

# 2. Configuration parameter
shape = (28, 28)  # origianl image size
N = 1000  # image sample size
p = 28*28 #= 784
o_img_size = 28
# n_comp = 10 # Number of compoents/archetype points
n_comp = 4 # Number of compoents/archetype points

# 3. Run NMF Algorithm
nmf = NMF(n_comp, o_img_size, shape, N, p)
nmf.run(maxiter=30, delta = 1e-3, threshold=1e-3, c1=1, c2=1, verbose=True, oracle=False)
# nmf.run(maxiter=30, delta = 1e-3, threshold=1e-3, c1=1, c2=1, verbose=False, oracle=False)
# nmf.run(maxiter=30, delta = 1e-3, threshold=1e-3, c1=1, c2=1, verbose=True, oracle=False)

# 4. Plot the result
nmf.plot_result()

