"""Module for formulating a
:external+sklearn:py:class:`sklearn.ensemble.GradientBoostingRegressor` or a
:external+sklearn:py:class:`sklearn.ensemble.GradientBoostingClassifier`
into a PySCIPOpt Model.
"""

import numpy as np

from ..modelling import AbstractPredictorConstr
from ..modelling.gradient_boosting import (
    aggregate_estimator_outputs,
    create_aggregation_constraints,
    create_sklearn_tree_estimators,
)
from .skgetter import SKgetter


def add_gradient_boosting_regressor_constr(
    scip_model,
    gradient_boosting_regressor,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate gradient_boosting_regressor into scip_model.

    The formulation predicts the values of output_vars using input_vars
    according to gradient_boosting_regressor.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the predictor should be inserted.
    gradient_boosting_regressor : :external+sklearn:py:class:`sklearn.ensemble.GradientBoostingRegressor`
        The gradient boosting regressor to insert as predictor.
    input_vars : np.ndarray or list
        Decision variables used as input for gradient boosting regressor in model.
    output_vars : np.ndarray or list, optional
        Decision variables used as output for gradient boosting regressor in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    GradientBoostingConstr
        Object containing information about what was added to scip_model to formulate
        gradient_boosting_regressor.

    Note
    ----
    |VariablesDimensionsWarn|

    """
    return GradientBoostingConstr(
        scip_model,
        gradient_boosting_regressor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        False,
        **kwargs,
    )


def add_gradient_boosting_classifier_constr(
    scip_model,
    gradient_boosting_classifier,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate gradient_boosting_classifier into scip_model.

    The formulation predicts the values of output_vars using input_vars
    according to gradient_boosting_classifier.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the predictor should be inserted.
    gradient_boosting_classifier : :external+sklearn:py:class:`sklearn.ensemble.GradientBoostingClassifier`
        The gradient boosting classifier to insert as predictor.
    input_vars : np.ndarray or list
        Decision variables used as input for gradient boosting classifier in model.
    output_vars : np.ndarray or list, optional
        Decision variables used as output for gradient boosting classifier in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    GradientBoostingConstr
        Object containing information about what was added to scip_model to formulate
        gradient_boosting_classifier.

    Note
    ----
    |VariablesDimensionsWarn|

    """
    return GradientBoostingConstr(
        scip_model,
        gradient_boosting_classifier,
        input_vars,
        output_vars,
        unique_naming_prefix,
        True,
        **kwargs,
    )


class GradientBoostingConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained
    :external+sklearn:py:class:`sklearn.ensemble.GradientBoostingRegressor`
    with SCIP.

    |ClassShort|
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        classification,
        **kwargs,
    ):
        self.estimators_ = []
        self.classification = classification
        SKgetter.__init__(self, predictor, **kwargs)
        if self.classification:
            if predictor.n_classes_ <= 2:
                self.output_size = 1
            else:
                self.output_size = predictor.n_classes_
        else:
            # Gradient Boosting Regression Trees in SKLearn only work for 1-D predictions
            self.output_size = 1
        AbstractPredictorConstr.__init__(
            self, scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        if not self.classification:
            assert (
                self.output.shape[-1] == 1
            ), f"Output dimension of SK-learn gradient boosting regressor should be 1. {self.output.shape[-1]} != 1"

        n_samples = self.input.shape[0]
        outdim = self.output.shape[-1]

        # Create individual models for each estimator (decision tree)
        estimators, tree_vars = create_sklearn_tree_estimators(
            self.scip_model,
            self.predictor,
            self.input,
            n_samples,
            outdim,
            self.unique_naming_prefix,
            self.classification,
            gbdt_or_rf="gbdt",
            **kwargs,
        )
        for estimator in estimators:
            self.estimators_.append(estimator)
        self._created_vars.append(tree_vars)

        # Get the constant shift and learning rate used
        lr = self.predictor.learning_rate
        if not self.classification:
            constant = np.array([self.predictor.init_.constant_[0][0] for _ in range(outdim)])
        else:
            constant = self.predictor.init_.predict_log_proba(np.zeros((1, self.input.shape[-1])))[
                0
            ]

        # Add the appropriate constraints for the case of regression and classification
        aggregate_tree_output = aggregate_estimator_outputs(tree_vars, lr, constant, aggr="sum")
        created_vars, created_cons = create_aggregation_constraints(
            self.scip_model,
            aggregate_tree_output,
            self.output,
            n_samples,
            outdim,
            self.unique_naming_prefix,
            self.classification,
        )
        for created_var in created_vars:
            self._created_vars.append(created_var)
        for created_con in created_cons:
            self._created_cons.append(created_con)
