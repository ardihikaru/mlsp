import numpy as np
from numpy import linalg as LA
import os, os.path
from matplotlib.image import imread
from PIL import Image
import glob
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from pca_numpy import  *

for filename in glob.glob("four_dataset/*.jpg"):
    img = imread(filename)
    # print(type(img))

    data = pd.DataFrame(img)
    print(type(data))
    print(data.head())

    break

