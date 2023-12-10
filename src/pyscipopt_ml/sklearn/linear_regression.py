"""Module for inserting ordinary Scikit-Learn regression models into a PySCIPOpt `Model`.

The following linear models are tested and should work:
   - :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
   - :external+sklearn:py:class:`sklearn.linear_model.Ridge`
   - :external+sklearn:py:class:`sklearn.linear_model.Lasso`
   - :external+sklearn:py:class:`sklearn.linear_model.ElasticNet`

"""

from .base_regression import BaseSKlearnRegressionConstr


def add_linear_regression_constr(
    scip_model, linear_regression, input_vars, output_vars=None, unique_naming_prefix="", **kwargs
):
    """Formulate linear_regression as a constraint in a PySCIPOpt Model.

    The formulation predicts the values of output_vars using input_vars according to
    linear_regression.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The scip model where the predictor should be inserted.
    linear_regression : :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
    The linear regression to insert. It can be any of the following types:
         * :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
         * :external+sklearn:py:class:`sklearn.linear_model.Ridge`
         * :external+sklearn:py:class:`sklearn.linear_model.Lasso`
         * :external+sklearn:py:class:`sklearn.linear_model.ElasticNet`
    input_vars: list or dict
        Decision variables used as input for the regression model
    output_vars: list or dict
        Decision variables used as output for the regression model
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    LinearRegressionConstr
        Object containing information about what was added to scip to formulate linear_regression.

    Note
    ----
    |VariablesDimensionsWarn|
    """
    return LinearRegressionConstr(
        scip_model, linear_regression, input_vars, output_vars, unique_naming_prefix, **kwargs
    )


class LinearRegressionConstr(BaseSKlearnRegressionConstr):
    """Class to model trained
    :external+sklearn:py:class:`sklearn.linear_model.LinearRegression` with SCIP

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
        if predictor.coef_.ndim > 1:
            self.output_size = predictor.coef_.shape[0]
        else:
            self.output_size = 1
        BaseSKlearnRegressionConstr.__init__(
            self,
            scip_model,
            predictor,
            input_vars,
            output_vars,
            unique_naming_prefix,
            **kwargs,
        )

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to SCIP."""
        self.add_regression_constr()
