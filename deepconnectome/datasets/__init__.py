"""
The :mod:`deepconnectome.datasets` module includes utilities to generate synthetic
network data.
"""

from ._sample_generator import make_regression
from ._sample_generator import make_classification

__all__ = ['make_regression',
           'make_classification'
]