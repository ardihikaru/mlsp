# 1. Import custom libraries
from hw3.libs.algorithms.pca import PCA

# 2. Setup the configuration
img_size = 28
n_dataset = 1000
# img_dataset_path = "./hw3/blend_4/"
img_dataset_path = "./hw3/mixture_dataset(0147)/"

# 3.v1 Calculate PCA with single n_comp
# n_comp = 2
# pca = PCA(img_size, n_dataset, n_comp, img_dataset_path)
# pca.run()

# 3.v2 Calculate PCA with multiple n_comp
# n_comps = [2, 16, 64, 256]
# pca = PCA(img_size, n_dataset, n_comps, img_dataset_path)
# pca.run_multiple()

# 3.v3 Calculate PCA with multiple n_comp and multiple input index
input_index = [0, 1, 2, 3] # Use the first 4 images to check the results
n_comp = 10
pca = PCA(img_size, n_dataset, n_comp, img_dataset_path, input_index)
pca.run()
