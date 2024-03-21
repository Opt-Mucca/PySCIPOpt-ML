"""
PySCIPOpt Machine learning
==================================

A Python package to help automatically formulate and embed trained machine learning models in
mathematical optimization models. The package supports a variety of regression / classification machine learning models
(linear, logistic, neural networks, decision trees, random forests,...) trained by
different machine learning frameworks (scikit-learn, PyTorch, LightGBM, XGBoost).

See #TODO: Insert readthedocs link here
for documentation.
"""

# read version from installed package

from ._version import __version__
from .add_predictor import add_predictor_constr
from .registered_predictors import sklearn_convertors
