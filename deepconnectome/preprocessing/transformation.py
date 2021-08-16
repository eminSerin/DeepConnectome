"""
The :mod:`deepconnectome.processing.transformation` module includes functions
to transform coefficients or input matrices. Coefficient transformation (i.e. Haufe transformation)
is only supported for generic linear machine learning algorithms such as Linear Regression, Liner SVM.
"""

import numpy as np
from deepconnectome.preprocessing.utils import ref_harmonic, ref_kullback, \
    ref_euclidian, ref_log_euclidian, shrink_to_group, is_positive_definite, \
    closest_symmetric, ensure_dims
from sklearn.base import TransformerMixin
from sklearn.covariance import ledoit_wolf, oas


def haufe_transformation(X, coef):
    """ Performs Haufe transformation (Haufe et al., 2014) on model coefficients.

    Model coefficients are filter values reflecting how features are interacted
    to each other to generate given outcome. Thus, due to noise and covariance,
    it might not imply the direct relationship between features and outcome.
    Thus, coefficients should be transformed using Haufe transformation.

    Note that, this function can only be used with linear generic machine
    learning models (e.g. Linear SVM, Linear Regression, Elastic Net).

    Args:
        X (numpy.ndarray): input matrix.
        coef (numpy.ndarray): coefficients derived from the trained
            model (e.g. clf.coef_).

    Returns:
        (numpy.ndarray): Transformed coefficients (i.e. activation patterns).

    Raises:
        ValueError: if coefficient matrix has unknown shape or is more than
            2 dimensional.

    References:
        Haufe, Stefan, et al. "On the interpretation of weight vectors of
            linear models in multivariate neuroimaging." Neuroimage 87 (2014): 96-110.
    """
    # Make sure that coef is an 1D array.
    n_sample = X.shape[-1]
    coef_dim = coef.shape
    if len(coef_dim) > 2:
        raise ValueError("coef variable must be maximum 2D matrix.")
    else:
        if len(coef_dim) == 2:
            if (coef_dim[-1] == n_sample) and coef_dim[0] == 1:
                coef = coef.squeeze()
            else:
                raise ValueError("coef with unknown shape!")

    # Compute activation patterns.
    cov_mat = np.cov(X.dot(coef.T), rowvar=False)
    norm_cov_mat = (coef / cov_mat) / (X.shape[-1] - 1)
    return X.T.dot(X.dot(norm_cov_mat.T))


def _find_nearest_positive_definite(X):
    """ Find the nearest positive-definite matrix to input.

    A python port of the code written by John D'Errico.

    Args:
        X (numpy.ndarray): Input matrix.

    Returns:
        (numpy.ndarray): Nearest positive definite matrix to the
            input matrix.

    References:
        N.J. Higham, "Computing a nearest symmetric positive semidefinite
            matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    """

    # Check input matrix is symmetric,
    # otherwise, find the closest symmetric matrix.
    X_sym = closest_symmetric(X)

    # Take eigen-decomposition of the symmetric X matrix.
    _, sigma, V = np.linalg.svd(X_sym)

    # Extract diagonal sigma matrix and
    # compute symmetric polar factor, which is SPD.
    symmetric_polar = V.T.dot(np.diag(sigma).dot(V))
    X_pos_def = (X_sym + symmetric_polar) / 2

    # Make sure that the transformed matrix is symmetric
    X_pos_def = closest_symmetric(X_pos_def)

    if is_positive_definite(X_pos_def):
        return X_pos_def

    # Tweak matrix a little bit to make it positive definite
    # if transformed matrix is still not positive definite.
    # It adds tiny multiple of identity matrix.
    spacing = np.spacing(np.linalg.norm(X))
    identity_mat = np.eye(X.shape[0])
    k = 1
    while not is_positive_definite(X_pos_def):
        min_eig = np.min(np.real(np.linalg.eigvals(X_pos_def)))
        X_pos_def += identity_mat * (-min_eig * k ** 2 + spacing)
        k += 1

    return X_pos_def


# Transforming matrices into nearest positive definite matrix
def transform_to_positive_definite(matrices, verbose=False):
    """Transforms input matrix to the nearest positive definite matrix.

    Args:
        matrices (numpy.ndarray): Input matrix
        verbose (bool): Whether to print progress.

    Returns:
        (numpy.ndarray): Positive definite matrices.
    """
    matrices_pd = np.empty_like(matrices)
    for i, x in enumerate(matrices):
        matrices_pd[i] = _find_nearest_positive_definite(x)
        if verbose:
            if i % 100 == 0:
                print(f"\rTransforming matrices {i}/{len(matrices_pd)} to positive definite...",
                      end="", flush=True)
    if verbose:
        print(f"\rMatrices were successfully transformed to positive definite.\n",
              end="", flush=True)
    return matrices_pd


class TangentSpace(TransformerMixin):
    """ Transforms positive definite covariance matrices
    into tangent space.

    Transforming covariance matrices (or positive definite
    correlation matrices) into tangent space has been shown to
    provide significantly higher prediction performance
    (Dadi et al., 2019; Pervaiz et al., 2020).

    Args:
        ref (str): Method to compute mean reference matrix.
            Options are: 'euclidian', 'log_euclidian', 'harmonic', 'kullback'
        shrink (str): Shrinkage estimator to shrink covariance or tangent
            projected matrices. Options are: 'ledoit_wolf', 'oas',
            'shrink_to_group'
        verbose (bool): Whether to print progress.

    Returns:
        (numpy.ndarray): Matrices projected to tangent space.

    References:
        Dadi, Kamalaker, et al. "Benchmarking functional connectome-based predictive models for resting-state fMRI."
            NeuroImage 192 (2019): 115-134.
        Pervaiz, Usama, et al. "Optimising network modelling methods for fMRI." Neuroimage 211 (2020): 116604.
    """
    def __init__(self, ref="euclidian", shrink=None, verbose=False):
        ref_opts = {'euclidian': ref_euclidian,
                    'log_euclidian': ref_log_euclidian,
                    'harmonic': ref_harmonic,
                    'kullback': ref_kullback}
        shrink_opts = {'ledoit_wolf': ledoit_wolf,
                       'oas': oas,
                       'shrink_to_group': shrink_to_group}
        self.ref_func = ref_opts[ref]
        self.shrink = shrink
        self.verbose = verbose

        if shrink is not None:
            if shrink in shrink_opts:
                self._shrink_func = shrink_opts[shrink]
            else:
                raise ValueError('Shinkage estimator is not found!')

    @ensure_dims
    def _check_positive_definite(self, X):
        """Checks whether input matrices are positive definite."""
        if len(X.shape) == 2:
            return is_positive_definite(X)
        for cov in X:
            if not is_positive_definite(cov):
                raise ValueError('Input matrix is not positive definite!')

    def _map_shrinkage(self, X, shrinkage_estimator='ledoit_wolf'):
        """Applies shrinkage each covariance matrix in X dataset."""
        if shrinkage_estimator not in ('ledoit_wolf', 'oas'):
            raise ValueError('Shinkage estimator parameter is not found!')
        shrinkage_func = eval(shrinkage_estimator)
        return np.array([shrinkage_func(cov)[0] for cov in X])

    def _form_symmetric_matrix(self, X, func=None):
        """Constructs a symmetric matrix from a given X matrix using
        eigenvalues and eigenvectors computed from the input matrix.
        While constructing, it also applies a custom function to eigenvalues
        to transform the matrix space."""
        if func is None:
            func = lambda x: 1. / np.sqrt(x)
        eigenvalues, eigenvectors = np.linalg.eigh(X)  # EVD on reference mean covariance matrix
        return np.dot(eigenvectors * func(eigenvalues), eigenvectors.T)

    def fit(self, X, y=None):
        """Fits the tangent space transformer to the given X matrix.

        Args:
            X (numpy.ndarray): Input matrix.
            y (None): Placeholder variable; no effect on function.

        Returns:
            self: TangentSpace instance.
                The object itself. Useful for chaining operations.
        """
        self._check_positive_definite(X)
        if hasattr(self, "_shrink_func"):
            X = self._shrink_func(X)
        self._ref_mean = self.ref_func(X)
        self._whitening = self._form_symmetric_matrix(self._ref_mean)
        return self

    def transform(self, X, y=None):
        """ Projects the input covariance matrices into tangent space
        using computed reference mean matrix.

        Args:
            X (numpy.ndarray): Input matrix.
            y (None): Placeholder variable; no effect on function.

        Returns:
            (numpy.ndarray): Transformed matrix.
        """
        self._check_positive_definite(X)
        X_trans = np.zeros_like(X)
        for i, mat in enumerate(X):
            X_trans[i] = self._form_symmetric_matrix(
                self._whitening.dot(mat).dot(self._whitening), np.log)

        if self.shrink == 'shrink_to_group':
            return shrink_to_group(X_trans, self._ref_mean)
        return X_trans

    def inverse_transform(self, X_trans, y=None):
        """ Inverse transform from tangent space to covariance matrix (Riemannian space).

        It does not work if shrinkage estimator has been used.

        Args:
            X_trans (numpy.ndarray): Transformed matrix (i.e. matrices in tangent space).
            y (None): Placeholder variable; no effect on function.

        Returns:
            (numpy.ndarray): Inverse transformed, covariance matrices.
        """
        if self.shrink is not None:
            raise NotImplementedError('Inverse transformation for shrunk covariance matrix has not been implemented!')

        sqrt_whitening = self._form_symmetric_matrix(self._ref_mean, lambda x: np.sqrt(x))
        X_untrans = np.zeros_like(X_trans)

        for i, mat in enumerate(X_trans):
            X_untrans[i] = sqrt_whitening.dot(
                self._form_symmetric_matrix(mat, np.exp)).dot(sqrt_whitening)
        return X_untrans
