# 1. Import custom libraries
from hw3.libs.algorithms.pca import PCA

# 2. Setup the configuration
img_size = 28
n_dataset = 982

# 3.v1 Calculate PCA with single n_comp
n_comp = 2
pca = PCA(img_size, n_dataset, n_comp)
pca.run()

# 3.v2 Calculate PCA with single n_comp
# n_comps = [2, 16, 64, 256]
# pca = PCA(img_size, n_dataset, n_comps)
# pca.run_multiple()
