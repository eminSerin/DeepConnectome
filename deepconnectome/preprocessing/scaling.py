"""
The :mod:`deepconnectome.processing.scaling` module includes functions
to scale feature space.
"""

from sklearn import preprocessing
from sklearn.base import TransformerMixin, clone


class ScaleTensorDec(TransformerMixin):
    """Decorator class that generalizes scaling classes from scikit-learn
    to work with 4D tensor data."""
    def __init__(self, cls):
        self._cls = cls
        self._scalers = []
        self._dims = None

    def __call__(self, *args, **kwargs):
        self._cls = self._cls(*args, **kwargs)
        self._scalers = []
        return self

    def fit(self, X, y=None, *args, **kwargs):
        self._dims = X.shape
        if len(self._dims) == 4:
            for c in range(self._dims[1]):
                X_2d = X[:, c, :, :].reshape(-1, self._dims[2] * self._dims[3])
                scaler = clone(self._cls)
                scaler.fit(X_2d, y, *args, **kwargs)
                self._scalers.append(scaler)
        elif len(self._dims) == 2:
            self._cls.fit(X, y)
        else:
            raise ValueError('Input array must be 4D or 2D.')
        return self

    def transform(self, X):
        if len(self._dims) == 4:
            for c in range(self._dims[1]):
                X_2d = X[:, c, :, :].reshape(-1, self._dims[2] * self._dims[3])
                X[:, c, :, :] = self._scalers[c].transform(X_2d).reshape(-1, self._dims[2], self._dims[3])
            return X
        elif len(self._dims) == 2:
            return self._cls.transform(X)
        else:
            raise ValueError('Input array must be 4D or 2D. ')


@ScaleTensorDec
class MinMaxScaler(preprocessing.MinMaxScaler):
    """MinMaxScaler from scikit-learn."""
    pass


@ScaleTensorDec
class RobustScaler(preprocessing.RobustScaler):
    """RobustScaler from scikit-learn."""
    pass


@ScaleTensorDec
class MaxAbsScaler(preprocessing.MaxAbsScaler):
    """MaxAbsScaler from scikit-learn."""
    pass


@ScaleTensorDec
class StandardScaler(preprocessing.StandardScaler):
    """StandardScaler from scikit-learn."""
    pass


@ScaleTensorDec
class QuantileTransformer(preprocessing.QuantileTransformer):
    """QuantileTransformer from scikit-learn."""
    pass


@ScaleTensorDec
class Normalizer(preprocessing.Normalizer):
    """Normalizer from scikit-learn."""
    pass
