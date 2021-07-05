"""
The :mod:`deepconnectome.models.utils` module includes helper classes and functions
required to construct machine learning and deep learning models.
"""

from torch import nn


class _Flatten(nn.Module):
    """Implementation of a class that flattens the input to
    be compatible with dense layers."""
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)



