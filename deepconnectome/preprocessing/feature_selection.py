"""
The :mod:`deepconnectome.processing.feature_selection` module includes functions
to select relevant features. It only works for generic machine learning algorithms.
Deep Neural Network algorithms performs feature selection inside their network.
"""

import numpy as np
from scipy import stats
from deepconnectome.preprocessing.utils import ConnectedComponent, matrix_stats_decorator
from sklearn.feature_selection import GenericUnivariateSelect, f_classif, RFE
from sklearn.utils.validation import check_is_fitted


class GenericFilterSelect(GenericUnivariateSelect):
    """Wrapper class for :class:`sklearn.feature_selection.GenericUnivariateSelect`"""
    def __init__(self, score_func=f_classif, *, mode='fpr', param=0.01):
        super().__init__(score_func=score_func, mode=mode, param=param)
        pass


class RecursiveFeatureElimination(RFE):
    """Wrapper class for :class:`sklearn.feature_selection.RFE`"""
    pass


class GraphFilterSelect(GenericUnivariateSelect):
    """Implementation of graph based filter feature selection (Serin et al,. nd).

    It performs desired feature selection using :class:`sklearn.feature_selection.GenericUnivariateSelect`
    to identify a set of relevant features (i.e. edges). Then, the largest connected component
    that may present in the set of selected edges is identified.
    GraphFilterSelect returns the vectorized edge indices of features present in
    the largest connected component.

    Attributes:
        vectorizer (deeepconnectome.io.input.MatrixVectorizer): :class:`deepconnectome.io.input.MatrixVectorizer`
            object used to vectorize input matrix.

    References:
        Serin, E., Zalesky, A., Matory, A., Walter, H. & Kruschwitz, J., (n.d.). NBS-Predict:
            A Prediction-based Extension of the Network-based Statistic. NeuroImage, (under review).

    """
    def __init__(self, score_func=f_classif, vectorizer=None,
                 mode='fpr', param=0.01):
        """Initializes class.

        Args:
            score_func (callable): Function taking X and y parameters and
                performing statistical analysis and returning a pair of arrays (stats, p-values).
            vectorizer (deeepconnectome.io.input.MatrixVectorizer): :class:`deepconnectome.io.input.MatrixVectorizer`
                object used to vectorize input matrix.
            mode (str): Feature selection mode. {'percentile', 'k_best', 'fpr', 'fdr', 'fwe'},
                Default to fpr.
            param (:obj:`int`, :obj:`float`): Threshold to select features based on statistics.

        See Also:
            :class:`deepconnectome.io.input.MatrixVectorizer`
        """
        super().__init__(score_func=score_func, mode=mode, param=param)
        if vectorizer is None:
            raise ValueError('You need to provide MatrixVectorizer object that '
                             'you used to vectorize the input adjacency matrices.')
        self.vectorizer = vectorizer

    def _get_support_mask(self):
        """Returns indices of selected features."""
        check_is_fitted(self)

        selector = self._make_selector()
        selector.pvalues_ = self.pvalues_
        selector.scores_ = self.scores_
        init_mask = np.where(selector._get_support_mask())[0]

        # Construct adjacency matrix of selected edges.
        n_nodes = self.vectorizer._dims[-1]
        mask_edge_idx = self.vectorizer.find_mat_idx(init_mask)
        adj_sel = np.zeros((n_nodes, n_nodes))
        for e in mask_edge_idx:
            adj_sel[e] = 1
        adj_sel += adj_sel.T

        # Find the largest connected component.
        comp = ConnectedComponent(n_conn_comp=1)

        # Return vectorized indices of relevant edges from the largest
        # connected component.
        return self.vectorizer.transform(comp.extract_sparse_matrices(adj_sel)[0]).astype(bool)


@matrix_stats_decorator
def pearson_r(X, y):
    """Wrapper function for :func:`scipy.stats.pearsonr`
    It can run on matrix"""
    return stats.pearsonr(X, y)


@matrix_stats_decorator
def spearman_r(X, y):
    """Wrapper function for :func:`scipy.stats.spearmanr`
    It can run on matrix"""
    return stats.spearmanr(X, y)


@matrix_stats_decorator
def point_biserial_r(X, y):
    """Wrapper function for :func:`scipy.stats.pointbiserialr`
    It can run on matrix"""
    return stats.pointbiserialr(X, y)


def t_test(X, y, **kwargs):
    """Wrapper function for :func:`scipy.stats.ttest_ind`
    It can run on matrix"""
    if len(np.unique(y)) > 2:
        raise ValueError('y must binary array!')
    mask = y == np.unique(y)[0]
    return stats.ttest_ind(X[mask], X[~mask], axis=0, **kwargs)
