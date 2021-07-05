"""
The :mod:`deepconnectome.models.ml` module includes classes that inherits from
the Classifier and Regressor classes from scikit-learn.
"""

import sklearn


# Construct Classifiers
class LogisticRegression(sklearn.linear_model.RidgeClassifier):
    """Wrapper for Ridge Regression Classifier"""
    pass


class ElasticNetClassifier(sklearn.linear_model.SGDClassifier):
    """Wrapper for Elastic Net Classifier.
    default l1_ratio = 0.5"""

    def __init__(self):
        super().__init__()
        self.penalty = 'elasticnet'
        self.l1_ratio = 0.5


class LinearSVC(sklearn.svm.LinearSVC):
    """Wrapper for Linear Support Vector Classifier that uses liblinear backend."""
    pass


class NonLinearSVC(sklearn.svm.NuSVC):
    """Wrapper for Nu SVC with rbf kernel."""
    pass


class GPC(sklearn.gaussian_process.GaussianProcessClassifier):
    """Wrapper for Gaussian Process Classifier"""
    pass


# Construct Regressors
class LinearRegression(sklearn.linear_model.Ridge):
    """Wrapper for Ridge Regressor"""
    pass


class ElasticNetRegressor(sklearn.linear_model.ElasticNet):
    """Wrapper for Elastic Net Regressor"""
    pass


class LinearSVR(sklearn.svm.LinearSVR):
    """Wrapper for Linear Support Vector Regressors that uses liblinear backend."""
    pass


class NonLinaerSVR(sklearn.svm.NuSVR):
    """Wrapper for NU SVR with rbf kernel"""
    pass


class GPR(sklearn.gaussian_process.GaussianProcessRegressor):
    """Wrapper for Gaussian Process Regressor"""
    pass
