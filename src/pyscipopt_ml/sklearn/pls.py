"""Module for formulating simple Scikit-Learn Partial Least Squares models in a PySCIPOpt model."""
import numpy as np

from ..modelling import AbstractPredictorConstr
from .skgetter import SKgetter


def add_pls_regression_constr(
    scip_model, pls_regression, input_vars, output_vars=None, unique_naming_prefix="", **kwargs
):
    """Formulate pls_regression in scip_model.

    The formulation predicts the values of output_vars using input_vars
    according to pls_regression.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The PySCIPOpt model where the predictor will be inserted.
    pls_regression : :external+sklearn:py:class:`sklearn.cross_decomposition.PLSRegression` or
        :external+sklearn:py:class:`sklearn.cross_decomposition.PLSCanonical`
        The partial least squares model to insert.
    input_vars : :list or dict
        Decision variables used as input for partial least squares regression in model.
    output_vars : list or dict
        Decision variables used as output for partial least squares regression in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    PLSRegressionConstr
        Object containing information about what was added to scip_model to
        formulate pls_regression.

    Note
    ----
    |VariablesDimensionsWarn|

    """

    return PLSRegressionConstr(
        scip_model, pls_regression, input_vars, output_vars, unique_naming_prefix, **kwargs
    )


class PLSRegressionConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained
    :external+sklearn:py:class:`sklearn.cross_decomposition.PLSRegression` or
    :external+sklearn:py:class:`sklearn.cross_decomposition.PLSCanonical` with SCIP

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
        self.output_size = predictor.n_components
        SKgetter.__init__(self, predictor, **kwargs)
        AbstractPredictorConstr.__init__(
            self,
            scip_model,
            input_vars,
            output_vars,
            unique_naming_prefix,
            **kwargs,
        )

    def add_regression_constr(self):
        """Add the prediction constraints to SCIP."""

        # Access the PLS model data
        x_mean = self.predictor._x_mean
        x_std = self.predictor._x_std
        coefs = self.predictor.coef_.T
        intercept = self.predictor.intercept_

        # Get the dimensions of the input and output
        n_samples = self.input.shape[0]
        outdim = self.output.shape[-1]

        # Create the PLS model constraints
        pls_cons = np.zeros((n_samples, outdim), dtype=object)
        rhs = (((self.input - x_mean[np.newaxis, :]) / x_std[np.newaxis, :]) @ coefs) + intercept
        for i in range(n_samples):
            for j in range(outdim):
                name = self.unique_naming_prefix + f"pls_reg_{i}_{j}"
                pls_cons[i][j] = self.scip_model.addCons(self.output[i][j] == rhs[i][j], name=name)

        self._created_cons.append(pls_cons)

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to SCIP."""
        self.add_regression_constr()
