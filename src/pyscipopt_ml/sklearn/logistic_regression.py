"""Module for formulating a
:external+sklearn:py:class:`sklearn.linear_model.LogisticRegression` in a
PySCIPOpt Model.
"""
import numpy as np
from pyscipopt import exp, quicksum

from ..exceptions import NoModel, ParameterError
from ..modelling.classification import argmax_bound_formulation
from ..modelling.var_utils import create_vars
from .base_regression import BaseSKlearnRegressionConstr


def add_logistic_regression_constr(
    scip_model,
    logistic_regression,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    output_type="classification",
    **kwargs,
):
    """Formulate logistic_regression in a SCIP Model.

    The formulation predicts the values of output_vars using input_vars according to
    logistic_regression.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The PySCIPOpt model where the predictor will be inserted.
    logistic_regression : :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression`
        The logistic regression model to insert.
    input_vars : :list or dict
        Decision variables used as input for logistic regression in model.
    output_vars : list or dict
        Decision variables used as output for logistic regression in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    output_type : {'classification', 'regression'}, default='classification'
        If the option chosen is 'classification' the output is 1 for exactly one class and
        0 for all others. If the option
        'regression' is chosen the output is the probability of each class.

    Returns
    -------
    LogisticRegressionConstr
        Object containing information about what was added to scip_model to formulate
        logistic_regression.

    Raises
    ------

    NoModel
        If the logistic regression is not a binary label regression

    ParameterError
        If the value of output_type is set to a non-conforming value (see above).

    Warning
    -------

    When there is a (near) tie for the most likely class, users should be aware that SCIP tolerances will allow
    either of the two classes to be selected. In a scenario where there are two classes, and the logistic regression
    model assigns probability of class 1 as 0.5000000001. Naturally this is larger than 0.4999999999 assigned to
    class 2, however from a numerics point of view in SCIP both can be selected.

    Note
    ----
    |VariablesDimensionsWarn|

    """
    return LogisticRegressionConstr(
        scip_model,
        logistic_regression,
        input_vars,
        output_vars,
        unique_naming_prefix,
        output_type,
        **kwargs,
    )


class LogisticRegressionConstr(BaseSKlearnRegressionConstr):
    """Class to model trained
    :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression` with SCIP

    |ClassShort|.
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars=None,
        unique_naming_prefix="",
        output_type="classification",
        **kwargs,
    ):
        if output_type not in ("classification", "regression"):
            raise ParameterError("output_type should be either 'classification' or 'regression'")

        self.output_type = output_type
        self.output_size = 1 if predictor.classes_.size <= 2 else predictor.classes_.size

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
        """Add the prediction constraints to the SCIP Model."""

        if self.input.ndim > 2 or self.output.ndim > 2:
            raise NoModel(
                self.predictor,
                "Logistic regression only supported for input.ndim <=2 and output.ndim <=2."
                f"Dim: {self.input.ndim} / {self.output.ndim}",
            )

        n_samples = self.input.shape[0]
        outdim = self.output.shape[-1]

        # First model the linear transformation
        affine_vars = create_vars(
            self.scip_model,
            shape=self.output.shape,
            vtype="C",
            lb=None,
            ub=None,
            name_prefix=self.unique_naming_prefix + "affine",
        )
        self._created_vars.append(affine_vars)
        self.add_regression_constr(output=affine_vars)

        # In the case of classification, we need to return binary results.
        if self.output_type == "classification":
            # Use an argmax formulation on the output of the affine transformation
            added_vars, added_cons = argmax_bound_formulation(
                self.scip_model,
                affine_vars,
                self.output,
                self.unique_naming_prefix,
                one_dim_center=0,
            )

            # Store all the appropriate variables
            for added_var in added_vars:
                self._created_vars.append(added_var)
            for added_con in added_cons:
                self._created_cons.append(added_con)
        else:
            # In this case we are turning probabilities and need to model the logistic function explicitly
            if outdim == 1:
                # In this case we are dealing with only a single class
                log_output_vars = create_vars(
                    self.scip_model,
                    shape=self.output.shape,
                    vtype="C",
                    lb=0,
                    ub=1,
                    name_prefix=self.unique_naming_prefix + "logistic",
                )
                logistic_cons = np.zeros(self.output.shape, dtype=object)
                output_eq_cons = np.zeros(self.output.shape, dtype=object)
                for i in range(n_samples):
                    for j in range(outdim):
                        name = self.unique_naming_prefix + f"sigmoid_{i}_{j}"
                        logistic_cons[i][j] = self.scip_model.addCons(
                            log_output_vars[i][j] == 1 / (1 + exp(-affine_vars[i][j])), name=name
                        )
                        name = self.unique_naming_prefix + f"output_eq_{i}_{j}"
                        output_eq_cons[i][j] = self.scip_model.addCons(
                            log_output_vars[i][j] == self.output[i][j], name=name
                        )
                self._created_vars.append(log_output_vars)
                self._created_cons.append(logistic_cons)
                self._created_cons.append(output_eq_cons)
            else:
                log_output_vars = create_vars(
                    self.scip_model,
                    shape=self.output.shape,
                    vtype="C",
                    lb=0,
                    ub=1,
                    name_prefix=self.unique_naming_prefix + "logistic",
                )
                # Get the actual predictive model used by sklearn in the logistic regression model
                ovr = self.predictor.multi_class in ["ovr", "warn"] or (
                    self.predictor.multi_class == "auto"
                    and (
                        self.predictor.classes_.size <= 2
                        or self.predictor.solver in ("liblinear", "newton-cholesky")
                    )
                )

                # If the model is one-vs-rest then model using logistic (sigmoid). Otherwise we use softmax
                cons_probability = np.zeros(self.output.shape, dtype=object)
                output_eq_cons = np.zeros(self.output.shape, dtype=object)
                for i in range(n_samples):
                    sum_logistic = quicksum(
                        1 / (1 + exp(-affine_vars[i][j])) for j in range(affine_vars[i].shape[0])
                    )
                    sum_softmax = quicksum(
                        exp(affine_vars[i][j]) for j in range(affine_vars[i].shape[0])
                    )
                    for j in range(outdim):
                        name = f"logistic_{i}_{j}"
                        if ovr:
                            logistic = 1 / (1 + exp(-affine_vars[i][j]))
                            cons_probability[i][j] = self.scip_model.addCons(
                                logistic / sum_logistic == log_output_vars[i][j], name=name
                            )
                        else:
                            cons_probability[i][j] = self.scip_model.addCons(
                                exp(affine_vars[i][j]) / sum_softmax == log_output_vars[i][j],
                                name=name,
                            )
                        name = self.unique_naming_prefix + f"output_eq_{i}_{j}"
                        output_eq_cons[i][j] = self.scip_model.addCons(
                            log_output_vars[i][j] == self.output[i][j], name=name
                        )
                self._created_vars.append(log_output_vars)
                self._created_cons.append(cons_probability)
                self._created_cons.append(output_eq_cons)
