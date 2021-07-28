"""
The :mod:`deepconnectome.preprocessing.utils` module includes helper classes and functions
related to preprocessing.
"""

import numpy as np
import networkx as nx
from functools import wraps


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
