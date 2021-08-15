"""
The :mod:`deepconnectome.processing.transformation` module includes functions
to transform coefficients or input matrices. Coefficient transformation (i.e. Haufe transformation)
is only supported for generic linear machine learning algorithms such as Linear Regression, Liner SVM.
"""

import numpy as np


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


