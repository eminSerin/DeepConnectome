"""
The :mod:`deepconnectome.io.utils` module includes helper classes and functions
related to input-output.
"""

import numpy as np


def check_symmetric(X, rtol=1e-05, atol=1e-08):
    """Checks if input matrix is symmetric.

    Args:
        X (numpy.ndrray): Input matrix. The input matrix can be more than 2 dimensional.
            Input shape must be [...,n_nodes,n_nodes].
        rtol (float): Relative tolerance parameter.
        atol (float): Absolute tolerance parameter.

    Returns:
        bool: Returns True if X matrix is symmetric; False otherwise.
    """
    # Flip last two elements in dimension indices.
    dim_idx = list(range(X.ndim))
    dim_idx = dim_idx[:-2:1] + dim_idx[:-3:-1]
    return np.allclose(X, X.transpose(dim_idx), rtol=rtol, atol=atol)


def mat_to_vec(X, tri='upper', mask=None):
    """Converts symmetric matrix to vectors.

    Args:
        X (numpy.ndarray): Input matrix
        tri (str): Which triangular to extract (upper or lower).
        mask (numpy.ndarray): Specific mask to extract edges from matrix.
            The "(np.array)" is optional.

    Returns:
        (numpy.ndarray): Vector containing extracted edges from the input matrix.

    Raises:
        ValueError: if X matrix is not symmetric.
        ValueError: if wrong value for tri argument passed.
        ValueError: if X the number of columns (nodes) in X matrix and
            mask does not match.
    """
    # Check if input matrix is symmetric.
    if not check_symmetric(X):
        raise ValueError('Input matrix is not symmetric!')

    # Vectorize matrix
    if mask is None:
        if tri == 'upper':
            mask = np.triu(np.ones(X.shape[-2:]), k=1).astype(bool)
        elif tri == 'lower':
            mask = np.tril(np.ones(X.shape[-2:]), k=-1).astype(bool)
        else:
            raise ValueError('Tri argument must be upper or lower.')
    else:
        if mask.shape[-1] != X.shape[-1]:
            raise ValueError('Number of nodes detected in the input vector ',
                             'does not fit to the number of nodes in the mask!')

    return X[..., mask], mask, np.where(mask)


def vec_to_mat(X_vec, mask):
    """Converts vectors into matrix.

    Args:
        X_vec (numpy.ndarray): Vectorized input matrix.
        mask (numpy.ndarray): Mask used to extract edges from symmetric matrix.

    Returns:
        X (numpy.ndarray): Symmetric matrix.

    Raises:
        ValueError: if X the number of columns (nodes) in X matrix and
            mask does not match.
    """
    n_nodes = np.int((np.sqrt(8 * X_vec.shape[-1] + 1) - 1.) / 2 + 1)

    if mask.shape[-1] != n_nodes:
        raise ValueError('Number of nodes detected in the input vector ',
                         'does not fit to numbar of nodes in the mask!')

    X = np.zeros(X_vec.shape[:-1] + (n_nodes, n_nodes))
    X[..., mask] = X_vec  # embed vector.

    # Embed vector to other triangular.
    X.swapaxes(-1, -2)[..., mask] = X_vec

    # Fill diagonal with 1.
    diag_mask = mask.copy()
    diag_mask.fill(False)
    np.fill_diagonal(diag_mask, True)
    X[..., diag_mask] = 1
    return X


def find_flat_idx(idx, n_nodes, tri='upper', mask=None):
    """Find indices of edges of a symmetric matrix in its vectorized form.

    It takes (x,y) indices such as (6,83) and returns the equivalent
    of them in vectorized form of matrix such as (876).

    Args:
        idx (list): List of tuples containing row,column indices.
            e.g. [(6,12),(1,2)]
        n_nodes (int): Number of nodes in matrix from which you get input indices.
        tri (str): A specific triangular to find indices from. Note that, if you
            set a specific triangular, find_flat_idx() returns indices from this
            triangular regardless of whether the input indices are not in this
            triangular of matrix.
            Returns indices from the flattened full matrix if "tri" set to None.
        mask (numpy.ndarray): Mask can be used if a specific part of matrix
            is needed to be masked, while finding indices.

    Returns:
        (numpy.ndarray): Numpy array of indices.

    Raises:
        ValueError: if idx is not a list of tuples.
    """
    try:
        if type(idx[0]) != tuple:
            raise ValueError()
    except ValueError as err:
        raise ValueError('Index input must be list of tuples!') from err

    adj = np.zeros((n_nodes, n_nodes))
    for i in idx:
        adj[i] = 1
    if tri is None:
        return np.where(adj.flatten())
    return np.where(mat_to_vec(adj + adj.T, tri=tri, mask=mask)[0])
