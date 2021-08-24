"""
The :mod:`deepconnectome.preprocessing.deconfounding` module includes
functions to remove confound effect from data.
"""

import numpy as np
from sklearn.base import TransformerMixin
from sklearn import preprocessing


class ConfoundRegression(TransformerMixin):
    """ Removes variance explained by confounds from given data.

    Linear method is fitted on each feature in a given data using
    confounds as predictors. If desired, polynomials of confounds
    can also be used as predictors to remove non-linear effect of
    confounds from given data.

    Args:
        normalize (bool): Whether to normalize confounds before
            calculating coefficients or not.
        intercept (bool): Whether to stack an intercept vector
            to confound matrix.
        degree (int): Degrees of confound polynomials.

    Returns:
        (numpy.ndarray): Deconfounded data.

    Reference:
        Snoek, L., Mileti, S., & Scholte, H. S. (2019). How to control for
            confounds in decoding analyses of neuroimaging data. NeuroImage,
            184, 741-760.
    """

    def __init__(self, normalize=False, intercept=True, degree=1):
        self.normalize = normalize
        self.intercept = intercept
        self.degree = degree
        self.confound = None
        self._weights = None

    def fit(self, X, confound):
        """ Fits confound regressor to given X matrix
        to calculate GLM coefficients.

        Args:
            X (numpy.ndarray): Data matrix to be deconfounded.
            confound (numpy.ndarray): Confound matrix.

        Returns:
            self: ConfoundRegression instance.
                The object itself. Useful for chaining operations.

        Raises:
            ValueError: if X and confound have different number of observations.

        """
        if confound.shape[0] != X.shape[0]:
            raise ValueError('X and confound have different number of observations!')

        self.confound = confound

        if self.normalize:
            scaler = preprocessing.MinMaxScaler()
            self.confound = scaler.fit_transform(self.confound)

        if self.degree > 1:
            self.confound = preprocessing.PolynomialFeatures(
                degree=self.degree,
                order='F').fit_transform(self.confound.reshape(-1, 1))

        if self.intercept:
            self.confound = np.c_[np.ones(self.confound.shape[0]),
                                  self.confound]

        self._weights = np.linalg.lstsq(self.confound, X, rcond=None)[0]
        return self

    def transform(self, X):
        """ Removes variance explained by the confounds.

        Args:
            X (numpy.ndarray): Input matrix.

        Returns:
            (numpy.ndarray): Deconfounded input matrix.

        Raises:
            ValueError: if input X matrix has different number of features
                from the one used in the fit method.

        """
        if X.shape[1] != self._weights.shape[1]:
            raise ValueError('X has different number of features!')

        return X - np.dot(self.confound, self._weights)
