"""
The :mod:`deepconnectome.datasets._sample_generator` module includes
functions to generate synthetic network data.
"""

# Import modules
import networkx as nx
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from deepconnectome.io.utils import make_triangular_idx


def _find_relevant_edges(G, n_edges, form_network=True):
    """ Returns indices of relevant edges that either forms a network
    (i.e. connected component) or are totally random.

    Args:
        G (networkx.classes.graph.Graph): Generated base graph.
        n_edges (int): Number of edges.
        form_network (bool): If form_network is True, relevant edges will form
            a network (i.e. connected component)

    Returns:
        list: List of indices of relevant edges.
    """
    n_nodes = G.number_of_nodes()
    if form_network:
        bfs_gen = nx.algorithms.traversal.bfs_edges(G, np.random.randint(n_nodes))
        return [next(bfs_gen) for _ in range(n_edges)]
    else:
        return [list(G.edges)[x] for x in np.random.choice(G.number_of_edges(),
                                                           n_edges, replace=False)]


def _gen_network(n_nodes, network='small_world', seed=None, *args, **kwargs):
    """ Generates synthetic network (e.g. small-world, scale-free or random).

    Args:
        n_nodes (int): Number of nodes
        network (str): Topology of network (small_world, scale_free, random_net).
        seed (int): Number for random number generator.
        *args: Additional arguments for particular network types.
        **kwargs: Additional arguments for particular network types.

    Returns:
        networkx.classes.graph.Graph: Generated network.

    Raises:
        ValueError: If wrong network type is given.
    """
    if network == 'small_world':
        def small_world(k=20, p=0.5):
            return nx.generators.watts_strogatz_graph(n_nodes, k, p, seed)

        return small_world(*args, **kwargs)

    elif network == 'scale_free':
        def scale_free(m=12):
            return nx.generators.barabasi_albert_graph(n_nodes, m, seed)

        return scale_free(*args, **kwargs)

    elif network == 'random':
        def random_net(p=0.21):
            return nx.generators.fast_gnp_random_graph(n_nodes, p, seed)

        return random_net(*args, **kwargs)

    else:
        raise ValueError('Wrong network type!')


def _gen_base_data(n_sample, n_nodes, n_rel_edges, network, connected,
                   seed=None, *args, **kwargs):
    """Helper function to generate base network data
    containing only gaussian noise (mean=0, std=1)."""
    G = _gen_network(n_nodes, network, seed, *args, **kwargs)
    rel_edge_idx = make_triangular_idx(_find_relevant_edges(G, n_rel_edges, connected),
                                       tri='upper')
    # Generate data.
    X = np.zeros((n_sample, n_nodes, n_nodes))
    edge_weight = np.random.randn(G.number_of_edges(), n_sample)
    for i, e in enumerate(G.edges):
        X[:, e[0], e[1]] = edge_weight[i, :]
    X = X + X.transpose((0, 2, 1))
    X = X + np.array([np.eye(n_nodes, n_nodes)] * n_sample)  # add 1 to diagonal.
    return X, G, rel_edge_idx


def make_classification(n_sample=50, n_nodes=100, n_rel_edges=50, cnr=0.5,
                        network='small_world', connected=True, seed=None,
                        *args, **kwargs):
    """Synthetic network data for simple binary classification.

    This function is the implementation of the data generation method presented in
    Zalesky et al., 2010. In addition to the presented method, make_classification
    can generate synthetic network with network topology resembling not only scale-free
    networks, but also small-world and random-networks.

    Args:
        n_sample (int): Number of samples.
        n_nodes (int): Number of nodes.
        n_rel_edges (int): Number of related edges (i.e. edges with ground truth).
        cnr (float): Contrast-to-noise ratio. Higher values result in higher classification performance.
        network (str): Topology of network (small_world, scale_free, random_net).
        connected (bool): If connected is True, relevant edges will form
            a network (i.e. connected component)
        seed (int): Number for random number generator.
        *args: Additional arguments for particular network types.
        **kwargs: Additional arguments for particular network types.

    Returns:
        X (numpy.array): 3D numpy array representing weighted adjacency matrix of
            generated synthetic networks.
        y (numpy.array): Numpy array of target.
        rel_edge_idx (list): List of coordinates of relevant edges (i.e. edges with ground truth).

    References:
        Zalesky, A., Fornito, A., & Bullmore, E. T. (2010). Network-based statistic:
            identifying differences in brain networks. NeuroImage, 53(4), 1197-1207.
    """
    X, G, rel_edge_idx = _gen_base_data(n_sample, n_nodes, n_rel_edges, network, connected, seed, *args,
                                        **kwargs)  # Generate base data.

    # Contrast
    contrast = np.zeros(n_sample)
    contrast[np.int(n_sample / 2):] = cnr
    X_contrast = np.zeros((n_sample, n_nodes, n_nodes))
    for i, e in enumerate(rel_edge_idx):
        X_contrast[:, e[0], e[1]] = contrast

    X = X + (X_contrast + X_contrast.transpose((0, 2, 1)))
    y = (contrast > 0) * 1
    return X, y, rel_edge_idx


def make_regression(n_sample=50, n_nodes=100, n_rel_edges=50, noise=0,
                    degree=1, network='small_world', connected=True,
                    seed=None, *args, **kwargs):
    """Synthetic network data for regression.

    This function is a generalization of sklearn.datasets.make_regression function,
    which embeds features in a network with given topology. The explanation of
    this function is given in Serin et al., (n.d.). It also allows for
    non-linear association between features and target, by adding polynomials
    (with given degree, e.g. 2) of relevant features before computing target variable.

    Args:
        n_sample (int): Number of samples.
        n_nodes (int): Number of nodes.
        n_rel_edges (int): Number of related edges (i.e. edges with ground truth).
        noise (float): Noise in target. Higher values result in lower regression performance.
        degree (int): Degree of polynomials of related edge values used when
            generating target variable. The association between data from
            related edges (i.e. features) is linear, if degree is 1,
            non-linear if it is higher. Higher degree values provide more difficult
            regression problems, thus result in lower regression performance.
        network (str): Topology of network (small_world, scale_free, random_net).
        connected (bool): If connected is True, relevant edges will form
            a network (i.e. connected component)
        seed (int): Number for random number generator.
        *args: Additional arguments for particular network types.
        **kwargs: Additional arguments for particular network types.

    Returns:
        X (numpy.array): 3D numpy array representing weighted adjacency matrix of
            generated synthetic networks.
        y (numpy.array): Numpy array of target.
        rel_edge_idx (list): List of coordinates of relevant edges (i.e. edges with ground truth).

    Raises:
        ValueError: If degree is less than 1.

    References:
        Serin, E., Zalesky, A., Matory, A., Walter, H. & Kruschwitz, J., (n.d.). NBS-Predict:
            A Prediction-based Extension of the Network-based Statistic. NeuroImage, (under review).
    """
    X, G, rel_edge_idx = _gen_base_data(n_sample, n_nodes, n_rel_edges, network, connected, seed, *args,
                                        **kwargs)  # Generate base data.

    # Gen. target from relevant edges.
    list_rel_edges = list(zip(*rel_edge_idx))
    X_rel_edges = X[:, list(list_rel_edges[0]), list(list_rel_edges[1])]
    if degree == 1:
        # Linear association between relevant features and target.
        coef = np.random.rand(n_rel_edges)
        y = X_rel_edges.dot(coef)
    elif degree > 1:
        # Add polynomials of relevant features before computing target.
        poly = PolynomialFeatures(degree=degree, order='F').fit_transform(X_rel_edges)
        coef = np.random.rand(poly.shape[1])
        y = poly.dot(coef)
    else:
        raise ValueError('Degree cannot be less than 1.')

    if noise > 0:
        y = np.random.randn(n_sample) * noise + y
    return X, y, rel_edge_idx
