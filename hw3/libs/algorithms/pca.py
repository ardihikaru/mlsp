import numpy as np
from matplotlib import pyplot as plt
from hw3.libs.common.dataset import ImgDataset

class PCA(ImgDataset):
    def __init__(self, img_size, n_dataset, n_comp, img_dataset_path=None):
        super().__init__(img_size, n_dataset, img_dataset_path)
        self.n_comp = n_comp

    """
    Step for PCA:
    1. Centering data
    2. Computing correlation matrix
    3. Eigendecomposition of correlation matrix
    4. Computing the projection matrix
    5. Computing principal components
    6. Reconstructing the data
    7. Computing reconstruction error
    8. Printing the graph
    """
    def run(self):
        centered_data, mean_vector = self.center_data()
        corr = self.correlation_matrix(centered_data)
        eigvals, eigvecs = self.eigen_decomposition(corr)

        proj_matrix = self.compute_projection_matrix(eigvals, eigvecs, self.n_comp)
        princ_comps = self.compute_principal_components(centered_data, proj_matrix)
        reconstructed_data = self.reconstruct_data(princ_comps, proj_matrix, mean_vector)

        self.error, self.err_var = self.compute_rec_error(self.data, reconstructed_data)
        print(" >>> Error = ", self.error)
        print("\n >>> err_var = ", self.err_var)

        self.plot_sample_result(self.data, reconstructed_data)

    def run_multiple(self):
        centered_data, mean_vector = self.center_data()
        corr = self.correlation_matrix(centered_data)
        eigvals, eigvecs = self.eigen_decomposition(corr)

        rec_data_list = []
        for i in range(0, len(self.n_comp)):
            n_comp = self.n_comp[i]
            proj_matrix = self.compute_projection_matrix(eigvals, eigvecs, n_comp)
            princ_comps = self.compute_principal_components(centered_data, proj_matrix)
            reconstructed_data = self.reconstruct_data(princ_comps, proj_matrix, mean_vector)
            rec_data_list.append(reconstructed_data)

            self.error, self.err_var = self.compute_rec_error(self.data, reconstructed_data)
            print(" >>> Error [%s] = %s" % (i, self.error))
            print("\n >>> err_var[%s] = %s" % (i, self.err_var))

        self.plot_sample_multi_result(self.data, rec_data_list)

    def get_error(self):
        return self.error

    def center_data(self):
        print('\nCentering data...')
        mean_vector = np.mean(self.data, axis=1).reshape(self.data.shape[0], 1)
        centered_data = self.data - mean_vector

        # if (globalv.print_results):
        #     print('Mean vector: \n', mean_vector)
        #     print('Centered data: \n', centered_data)
        return centered_data, mean_vector

    def correlation_matrix(self, centered_data):
        print('\nComputing correlation matrix...')
        # D = globalv.D
        # N = globalv.N

        correlation_matrix = np.zeros((self.D, self.D))
        for i in range(self.N):
            correlation_matrix += (centered_data[:, i].reshape(self.D, 1)).dot(centered_data[:, i].reshape(self.D, 1).T)
        correlation_matrix = correlation_matrix / self.N

        # if (globalv.print_results):
        #     print(correlation_matrix)

        return correlation_matrix

    def eigen_decomposition(self, corr):
        print('\nEigendecomposition of correlation matrix...')
        # D = globalv.D
        # N = globalv.N

        eigvals, eigvecs = np.linalg.eig(corr)

        for i in range(len(eigvals)):
            eigv = eigvecs[:, i].reshape(1, self.D).T
            np.testing.assert_array_almost_equal(corr.dot(eigv), eigvals[i] * eigv, decimal=6, err_msg='', verbose=True)

        # if (globalv.print_results):
        #     print('Eigenvalues: \n', eigvals)
        #     print('Eigenvectors: \n', eigvecs)

        return eigvals, eigvecs

    def compute_projection_matrix(self, eigvals, eigvecs, r):
        print('\nComputing the projection matrix...')
        # D = globalv.D
        # N = globalv.N

        # List of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eigvals[i]), eigvecs[:, i]) for i in range(len(eigvals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low (reverse=True)
        eig_pairs.sort(key=lambda tup: tup[0], reverse=True)

        # Projection matrix
        proj_matrix = np.zeros((self.D, r))
        for i in range(r):
            proj_matrix[:, i] = eig_pairs[i][1].reshape(1, self.D)

        # if (globalv.print_results):
        #     print('Projection matrix: \n', proj_matrix)

        return proj_matrix

    def compute_principal_components(self, centered_data, proj_matrix):
        print('\nComputing principal components...')
        p = (proj_matrix.T).dot(centered_data)

        # if (globalv.print_results):
        #     print('Principal components: \n', p)

        return p

    def reconstruct_data(self, princ_comps, proj_matrix, mean_vector):
        print('\nReconstructing the data...')
        reconstructed_data = (proj_matrix).dot(princ_comps) + mean_vector
        # reconstructed_data = proj_matrix.dot(princ_comps) + mean_vector # kudu ngene kan?

        # if (globalv.print_results):
        #     print('Reconstructed data: \n', reconstructed_data)

        return reconstructed_data

    def compute_rec_error(self, data, rec_data):
        print('\nComputing reconstruction error...')
        # D = globalv.D
        # N = globalv.N

        error = data - rec_data
        error_var = 0
        for i in range(self.N):
            error_var += np.linalg.norm(error[:, i]) ** 2
        error_var = error_var / self.N

        # if (globalv.print_results):
        #     print('Error: \n', error)
        # print('Error variance: ', error_var)

        return error, error_var

    def plot_sample_result(self, data, rec_data):
        # print(" ** DISINI ..")
        # print(data.shape)
        # print(data)
        ori_data = data[:, 0].reshape(self.img_size, self.img_size)
        rec_data = rec_data[:, 0].reshape(self.img_size, self.img_size)

        # print(len(ori_data))
        # print(len(ori_data[0]))
        # print(ori_data)

        f, (ax1, ax2) = plt.subplots(1, 2)
        f.suptitle('PCA comparison with #Dim = %s' % self.n_comp)
        ax1.imshow(ori_data, cmap='gray', interpolation='none')
        ax1.set_title('Original Image')
        ax2.imshow(rec_data, cmap='gray', interpolation='none')
        ax2.set_title('Compressed Image')
        plt.show()

    def plot_sample_multi_result(self, data, rec_data_list):
        # ori_data = data[:, 0].reshape(self.img_size, self.img_size)
        ori_data = data[:, 0].reshape(self.img_size, self.img_size)

        # fig4, axarr = plt.subplots(3,2,figsize=(8,8))
        fig4, axarr = plt.subplots(4, 2, figsize=(9, 9))
        for i in range(0, len(self.n_comp)):
            tmp_rec_data = rec_data_list[i][:, 0].reshape(self.img_size, self.img_size)
            axarr[i, 0].imshow(ori_data, cmap='gray', interpolation='none')
            axarr[i, 0].set_title('Original Image')
            axarr[i, 0].axis('off')
            axarr[i, 1].imshow(tmp_rec_data, cmap='gray', interpolation='none')
            axarr[i, 1].set_title('Compressed with n_comp = %s' % self.n_comp[i])
            axarr[i, 1].axis('off')
        plt.show()


