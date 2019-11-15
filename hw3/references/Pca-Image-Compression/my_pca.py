from normalization import Normalization
import scipy as sp
from scipy import linalg

class Pca:

    def __init__(self, data):
        self._normalization = Normalization(data)

        normalized_data = self._normalization.normalized_dataset()
        data_matrix = sp.matrix(normalized_data)
        m = data_matrix.shape[0]
        covariance_matrix = data_matrix.transpose() * data_matrix
        covariance_matrix /= m
        eig_decomp = linalg.eigh(covariance_matrix)
        self._n = len(eig_decomp[0])
        self._pcas = sp.zeros((self._n, self._n))
        for i in range(self._n):
            self._pcas[i, :] = eig_decomp[1][:, self._n - i - 1]

        self._eig_vals = list(eig_decomp[0])
        self._eig_vals.reverse()

    @property
    def pcas(self):
        return self._pcas

    @property
    def eig_vals(self):
        return self._eig_vals

    @property
    def n(self):
        return self._n

    def project(self, vector, k):
        v = self._normalization.normalize_x(vector)

        # project it
        v = sp.array(v)
        dot_product = lambda pca: sum(pca[j] * v[j] for j in range(len(v)))
        return [dot_product(self.pcas[i]) for i in range(k)]

    def deproject(self, vector):
        v = list(vector)
        result = sp.zeros(self._n)
        for i in range(len(v)):
            result += self._pcas[i] * v[i]

        result = self._normalization.denormalize_x(list(result))
        return list(result)