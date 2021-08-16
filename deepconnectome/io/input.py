"""
The :mod:`deepconnectome.io.input` module includes functions and classes
to handle input data.
"""

from deepconnectome.io.utils import vec_to_mat, find_flat_idx, mat_to_vec, find_mat_idx


class MatrixVectorizer:
    """Class to vectorize a symmetric matrix.
    It also allows for inverse vectorize (vector to matrix, inverse_transform()), and
    transform matrix indices to vector indices (find_flat_idx()).
    """

    def __init__(self, tri='upper'):
        """Initializes class.

        Args:
            tri (str): Triangular to extract edges from (upper or lower).
        """
        self.tri = tri

    def fit(self, X):
        """Returns vectorized matrix.

        Args:
            X (numpy.ndarray): Input matrix.

        Returns:
            (numpy.ndarray): Vectorized matrix.
        """
        return self._fit_transform(X)

    def transform(self, X, mask=None):
        """Transforms (vectorize) input matrix.

        Args:
            X (numpy.ndarray): Input matrix.
            mask (numpy.ndarray): Mask to extract edges from the input matrix.

        Returns:
            (numpy.ndarray): Vectorized matrix.

        """
        if mask is None:
            mask = self._mask
        return mat_to_vec(X, tri=self.tri, mask=mask)[0]

    def fit_transform(self, X):
        """Fit transform method."""
        return self._fit_transform(X)

    def inverse_transform(self, X, mask=None):
        """Converts vectorized matrix back to a symmetric matrix.

        Args:
            X (numpy.ndarray): Vectorized matrix.
            mask (numpy.ndarray): Mask used to extract edges from symmetric matrix.

        Returns:
            (numpy.ndarray): Symmetric matrix.

        """
        if mask is None:
            mask = self._mask
        return vec_to_mat(X, mask)

    def find_flat_idx(self, idx, n_nodes=None):
        """Transforms matrix indices to vector indices.

        Args:
            idx (list): List of tuples containing matrix indices (row,column).
            n_nodes (int): Number of nodes in symmetric matrix from which
                indices are collected.

        Returns:
            (list): List of vector indices.

        """
        if n_nodes is None:
            return find_flat_idx(idx, self._dims[-1],
                                 tri=self.tri, mask=self._mask)
        return find_flat_idx(idx, n_nodes, tri=self.tri)

    def find_mat_idx(self, idx, mask=None):
        """Transforms vectorized indices to matrix indices.

        Args:
            idx (list): List of vectorized indices.
            mask (numpy.ndarray): Mask used to extract edges from symmetric matrix.
                mask attribute stored in the object will be used if no mask is provided.

        Returns:
            (list): List of tuples containing matrix indices (row,column).
        """
        if mask is None:
            mask = self._mask
        return find_mat_idx(idx, mask=mask)

    def _fit_transform(self, X):
        """Private function used to transform metrix."""
        self._dims = X.shape

        # Vectorize matrix
        X_vec, self._mask, self._idx = mat_to_vec(X, self.tri)
        return X_vec
