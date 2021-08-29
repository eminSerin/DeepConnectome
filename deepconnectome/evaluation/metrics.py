"""
The :mod:`deepconnectome.evaluation.metrics` module includes
additional functions to evaluate models' performance that are
not available in the sklearn.metrics module. Common performance
metrics will be directly used from the sklearn.metrics module.
"""
import sklearn
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import dice


def pearson_score(y_true, y_pred):
    """Computes Pearson's Correlation Coefficient between
    true and predicted labels.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        (float): Pearson's correlation coefficient.
    """
    r, _ = pearsonr(y_true, y_pred)
    return r


def dice_score(y_true, y_pred):
    """Computers Dice score between true
    and predicted labels.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        (float): Dice score.
    """
    return dice(y_true, y_pred)


def evaluate_performance(y_true, y_pred, metric_func=None, *args, **kwargs):
    """Evaluates prediction performance using given
    metric function.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.array): Predicted labels.
        metric_func (function): Metric function.
        *args: Additional positional arguments to pass
            to metric function.
        **kwargs: Additional keyword arguments to pass
            to metric function.

    Returns:
        (float): Performance score.
    """
    if metric_func is None:
        is_y_nominal = (len(np.unique(y_true))/len(y_true)) < 0.01
        if is_y_nominal:
            metric_func = sklearn.metrics.accuracy_score
        else:
            metric_func = sklearn.metrics.explained_variance_score

    return metric_func(y_true, y_pred, *args, **kwargs)