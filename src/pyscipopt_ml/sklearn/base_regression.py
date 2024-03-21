"""Module for inserting simple Scikit-Learn regression models into a PySCIPOpt model.

All linear models should work:
   - :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
   - :external+sklearn:py:class:`sklearn.linear_model.Ridge`
   - :external+sklearn:py:class:`sklearn.linear_model.Lasso`
   - :external+sklearn:py:class:`sklearn.linear_model.ElasticNet`
"""
import numpy as np

from ..modelling import AbstractPredictorConstr
from .skgetter import SKgetter


class BaseSKlearnRegressionConstr(SKgetter, AbstractPredictorConstr):
    """Predict output variables using Linear Regression that takes PySCIPOpt variables as input.

    |ClassShort|.
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars=None,
        unique_naming_prefix="",
        **kwargs,
    ):
        SKgetter.__init__(self, predictor, **kwargs)
        AbstractPredictorConstr.__init__(
            self,
            scip_model,
            input_vars,
            output_vars,
            unique_naming_prefix,
            **kwargs,
        )

    def add_regression_constr(self, output=None):
        """Add the prediction constraints to SCIP."""

        if output is None:
            output = self.output

        coefs = self.predictor.coef_.T
        intercept = self.predictor.intercept_

        # Transform the input by an affine transformation (Ax + b)
        affine = self.input @ coefs + intercept
        if affine.shape != output.shape:
            affine = affine.reshape(output.shape)
        affine_cons = np.zeros(affine.shape, dtype=object)

        # Iterate through all entries and create the appropriate constraints
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                name = self.unique_naming_prefix + f"linreg_{i}_{j}"
                affine_cons[i][j] = self.scip_model.addCons(
                    affine[i][j] == output[i][j], name=name
                )

        self._created_cons.append(affine_cons)
