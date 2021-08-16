"""
The :mod:`deepconnectome.preprocessing.utils` module includes helper classes and functions
related to preprocessing.
"""

import numpy as np
import networkx as nx
from functools import wraps
from scipy import stats
from deepconnectome.io.utils import check_symmetric


class ConnectedComponent:
    """Extract adjacency matrices of connected component from given
    input adjacency matrix.

    Attributes:
        n_conn_comp (int): The number of connected component to return.
    """

    def __init__(self, n_conn_comp=1):
        self.n_conn_comp = n_conn_comp

    def extract_sparse_matrices(self, adj):
        """Extract sparse matrices of connected components from input
        adjacency matrix.

        The output adjacency matrices have the same shape with the input
        adjacency matrix. Edge values for non-connected nodes will be set
        to 0.

        Args:
            adj (numpy.ndarray): Input adjacency matrix.

        Returns:
            (:obj:`list` of :obj:`numpy.ndarray`): List of adjacency matrices of connected components.
        """
        adj_conn_comp_list = []
        conn_comps = sorted(nx.connected_components(nx.from_numpy_matrix(adj)),
                            key=len, reverse=True)
        for i in range(self.n_conn_comp):
            mask = np.full(adj.shape, False)
            mask[np.array(list(conn_comps[i])), :] = True
            mask[:, np.array(list(conn_comps[i]))] = True
            adj_conn_comp_list.append(mask * adj)
        return adj_conn_comp_list

    def extract_subgraph_matrices(self, adj):
        """Extract adjacency matrices of connected components from input
        adjacency matrix.

        The number of columns and rows in the output adjacency matrices is the
        number of nodes present in the connected components.

        Args:
            adj (numpy.ndarray): Input adjacency matrix.

        Returns:
            (:obj:`list` of :obj:`numpy.ndarray`): List of adjacency matrices of connected components.
        """
        G = nx.from_numpy_matrix(adj)
        conn_comps = sorted(nx.connected_components(G), key=len, reverse=True)
        return [nx.adj_matrix(G.subgraph(conn_comps[i])).toarray() for i in range(self.n_conn_comp)]


def matrix_stats_decorator(func):
    """Decorator function make statistics functions to run
    on 2D matrix."""

    @wraps(func)
    def wrapper(X, y):
        stat, p = np.array(list(zip(*[func(x, y) for x in X.T])))
        return stat, p

    return wrapper


def ensure_dims(func):
    """Decorator function to ensure that the input
    matrix is in either NxN or MxNxN shape."""

    @wraps(func)
    def wrapper(X, *args, **kwargs):
        X_dim = X.shape
        if (X_dim[-1] != X_dim[-2]) or len(X_dim) > 3:
            raise ValueError('Input array must have dimensions of NxN or MxNxN!')
        return func(X, *args, **kwargs)

    return wrapper


def is_positive_definite(X):
    """Checks if given matrix is positive definite.

    Args:
        X (numpy.ndarray): Input matrix.

    Returns:
        (bool): Whether input matrix is positive definite.

    References:
        https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """
    try:
        _ = np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False


@ensure_dims
def closest_symmetric(X):
    """Finds closest symmetric matrix to X matrix.

    Args:
        X (numpy.ndarray): Input matrix.

    Returns:
        X (numpy.ndarray): Symmetric matrix closest to the input matrix.
    """
    if not check_symmetric(X):
        return (X + X.T) / 2
    return X


@ensure_dims
def ref_euclidian(X):
    """ Takes euclidian mean of input matrix.

    Args:
        X (numpy.ndarray): Input matrix.

    Returns:
        (numpy.ndarray): Mean matrix.
    """
    return np.mean(X, axis=0)


@ensure_dims
def ref_log_euclidian(X):
    """Computes log euclidian mean of input matrix.

    Args:
        X (numpy.ndarray): Input matrix.

    Returns:
        (numpy.ndarray): Mean matrix.
    """
    return np.exp(np.nanmean(np.log(X), axis=0))


@ensure_dims
def ref_harmonic(X):
    """Computes harmonic mean of input matrix.

    Args:
        X (numpy.ndarray): Input matrix.

    Returns:
        (numpy.ndarray): Mean matrix.
    """
    return np.linalg.inv(np.mean(np.linalg.inv(X), axis=0))


@ensure_dims
def ref_kullback(X):
    """Computes Kullback mean of input matrix.

    Kullback mean is a geometric mean of euclidian and
    harmonic mean of input matrix.

    Args:
        X (numpy.ndarray): Input matrix.

    Returns:
        (numpy.ndarray): Mean matrix.

    References:
        Pervaiz, Usama, et al. "Optimising network modelling methods for fMRI." Neuroimage 211 (2020): 116604.
    """
    return np.nan_to_num(stats.mstats.gmean(np.array([ref_log_euclidian(X), ref_harmonic(X)]), axis=0), nan=0)


def shrink_to_group(tangent_mat, ref_mat, alpha=0.1):
    """Shrinkage estimator to shrink tangent matrix to
    group mean.

    Args:
        tangent_mat (numpy.ndarray): tangent matrix projected from covariance matrix.
        ref_mat (numpy.ndarray): group average reference matrix.
        alpha (numpy.float): shrinkage term. Higher values result in greater shrinkage
            to the group average.

    Returns:
        (numpy.ndarray): Shrunk tangent matrix.
    """
    return (1 - alpha) * tangent_mat + (alpha * ref_mat)
