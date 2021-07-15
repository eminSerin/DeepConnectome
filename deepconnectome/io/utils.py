"""
The :mod:`deepconnectome.io.utils` module includes helper classes and functions
related to input-output.
"""

import numpy as np


def _check_symmetric(X, rtol=1e-05, atol=1e-08):
    """Checks if input matrix is symmetric.

    Args:
        X (numpy.array): Input matrix.
        rtol (float): Relative tolerance parameter.
        atol (float): Absolute tolerance parameter.

    Returns:
        bool: Returns True if X matrix is symmetric; False otherwise.
    """
    return np.allclose(X, X.T, rtol=rtol, atol=atol)
