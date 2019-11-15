import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as LA
from matplotlib.image import imread

# 1. Import custom libraries
from hw3.libs.algorithms.ica import ICA

# 2. Configuration parameter
# o_img_size = 800
o_img_size = 28

# ica = ICA(o_img_size)
ica = ICA(o_img_size)
ica.run()
