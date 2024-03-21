"""Module for formulating a :external+sklearn:py:class:`sklearn.ensemble.RandomForestRegressor`
or :external+sklearn:py:class:`sklearn.ensemble.RandomForestClassifier` into a PySCIPOpt Model.
"""
import numpy as np

from ..modelling import AbstractPredictorConstr
from ..modelling.gradient_boosting import (
    aggregate_estimator_outputs,
    create_aggregation_constraints,
    create_sklearn_tree_estimators,
)
from .skgetter import SKgetter


def add_random_forest_regressor_constr(
    scip_model,
    random_forest_regressor,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate random_forest_regressor in scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    random_forest_regressor.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the predictor should be inserted.
    random_forest_regressor : :external+sklearn:py:class:`sklearn.ensemble.RandomForestRegressor`
        The random forest regressor to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for random forest in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for random forest in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    RandomForestConstr
       Object containing information about what was added to scip_model to formulate
       random_forest_regressor.

    Note
    ----
    |VariablesDimensionsWarn|
    """

    return RandomForestConstr(
        scip_model,
        random_forest_regressor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        False,
        **kwargs,
    )


def add_random_forest_classifier_constr(
    scip_model,
    random_forest_classifier,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate random_forest_classifier in scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    random_forest_classifier.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the predictor should be inserted.
    random_forest_classifier : :external+sklearn:py:class:`sklearn.ensemble.RandomForestClassifier`
        The random forest classifier to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for random forest in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for random forest in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    RandomForestConstr
       Object containing information about what was added to scip_model to formulate
       random_forest_classifier.

    Note
    ----
    |VariablesDimensionsWarn|
    """

    return RandomForestConstr(
        scip_model,
        random_forest_classifier,
        input_vars,
        output_vars,
        unique_naming_prefix,
        True,
        **kwargs,
    )


class RandomForestConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained :external+sklearn:py:class:`sklearn.ensemble.RandomForestRegressor`
    or :external+sklearn:py:class:`sklearn.ensemble.RandomForestClassifier` with SCIP

    |ClassShort|.
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
        if self.classification:
            if predictor.n_classes_ <= 2:
                self.output_size = 1
            else:
                self.output_size = predictor.n_classes_
        else:
            self.output_size = predictor.n_outputs_
        SKgetter.__init__(self, predictor, **kwargs)
        AbstractPredictorConstr.__init__(
            self, scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Predict output variables y from input variables X using the decision tree.

        Both X and y should be arrays or lists of variables of conforming dimensions.
        """

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
            gbdt_or_rf="rf",
            **kwargs,
        )
        for estimator in estimators:
            self.estimators_.append(estimator)
        self._created_vars.append(tree_vars)

        # Get the constant shift used
        constant = np.array([0 for _ in range(self.output.shape[-1])])

        # Add the appropriate constraints for the case of regression and classification
        aggregate_tree_output = aggregate_estimator_outputs(tree_vars, 1, constant, aggr="avg")
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
