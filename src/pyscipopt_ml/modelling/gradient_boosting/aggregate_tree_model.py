""" Utilities for modelling gradient boosting decision trees and random forest constraints """

import numpy as np

from ...sklearn.decision_tree import (
    add_decision_tree_classifier_constr,
    add_decision_tree_regressor_constr,
)
from ..base_predictor_constraint import AbstractPredictorConstr
from ..classification.argmax_model import argmax_bound_formulation
from ..decision_tree import leaf_formulation
from ..var_utils import create_vars


def aggregated_estimator_formulation(
    scip_model,
    _input,
    output,
    tree_vars,
    trees,
    constant,
    lr,
    n_estimators,
    unique_naming_prefix,
    epsilon,
    aggr,
    classification,
    **kwargs,
):
    """
    Creates the model that represents the aggregation of estimators into a single output.
    This function is used exclusively for the case where the estimators are decision trees, and the larger
    predictor is either a gradient boosting decision tree or random forest.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    _input : np.ndarray
        The input variables that are passed to each decision tree
    output : np.ndarray
        The output variables of the predictor
    tree_vars : np.ndarray
        The PySCIPOpt variables that have been created to represent the output of each decision tree (i.e. estimator)
    trees : list
        A list of lists containing dictionary information that completely describe each decision tree (i.e. estimator)
    constant : np.ndarray
        An array of constant shift values that are added to the output values of each decision tree (i.e. estimator)
    lr : float or int
        The learning rate used while training. For GBDT / RF this scales the output of each tree
    n_estimators : int
        The number of decision trees (i.e. estimators)
    unique_naming_prefix : str
        The unique naming prefix string that goes before all variables and constraints that are constructed by SCIP
    epsilon : float
        The epsilon that is used for each decision tree model. See
        :py:func:`pyscipopt_ml.modelling.decision_tree.leaf_formulation`.
    aggr : str, "sum" or "avg"
        The aggregation method used in the formulation. Either the estimators are averages or summed.
    classification : bool
        Whether the aggregated output of each decision tree (i.e. estimator) should be used for classification.

    Returns
    -------

    estimators : list
        A list of :py:class`pyscipopt_ml.modelling.aggregate_tree_model.TreeEstimator`
    created_vars : list
        A list containing all created PySCIPOpt vars
    created_cons : list
        A list containing all created PySCIPOpt cons
    """

    # Get the number of samples and output dimension
    n_samples = _input.shape[0]
    outdim = output.shape[-1]

    # Create the individual tree estimators
    estimators = create_tree_estimators(
        scip_model,
        _input,
        tree_vars,
        trees,
        n_estimators,
        outdim,
        unique_naming_prefix,
        epsilon,
        False,
        **kwargs,
    )

    # Aggregate the trees over the output dimension
    aggregate_tree_output = aggregate_estimator_outputs(tree_vars, lr, constant, aggr=aggr)

    # Formulate the appropriate constraints
    created_vars, created_cons = create_aggregation_constraints(
        scip_model,
        aggregate_tree_output,
        output,
        n_samples,
        outdim,
        unique_naming_prefix,
        classification,
    )

    return estimators, created_vars, created_cons


def create_aggregation_constraints(
    scip_model,
    aggregate_tree_output,
    output,
    n_samples,
    outdim,
    unique_naming_prefix,
    classification,
):
    """
    Creates the variables and constraints that link the output of the predictor itself and the aggregation of each
    estimator.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    aggregate_tree_output : np.ndarray
        The aggregated output variables of each decision tree
    output : np.ndarray
        The output variables of the predictor
    n_samples : int
        The number of samples
    outdim : int
        The number of outputs of each decision tree (i.e. estimator)
    unique_naming_prefix : str
        The unique naming prefix string that goes before all variables and constraints that are constructed by SCIP
    classification : bool
        Whether the aggregated output of each decision tree (i.e. estimator) should be used for classification.

    Returns
    -------
    created_vars : list
        A list containing all created PySCIPOpt vars
    created_cons : list
        A list containing all created PySCIPOpt cons
    """

    # Formulate the appropriate constraints
    created_cons = []
    created_vars = []
    if not classification:
        sum_tree_cons = np.zeros((n_samples, outdim), dtype=object)
        for i in range(n_samples):
            for j in range(outdim):
                name = unique_naming_prefix + f"tree_sum_{i}_{j}"
                sum_tree_cons[i][j] = scip_model.addCons(
                    output[i][j] == aggregate_tree_output[i][j], name=name
                )
        created_cons.append(sum_tree_cons)
    else:
        new_vars, new_cons = argmax_bound_formulation(
            scip_model, aggregate_tree_output, output, unique_naming_prefix, one_dim_center=0
        )  # TODO: Determine for which models is one_dim_center=0.5
        for added_var in new_vars:
            created_vars.append(added_var)
        for added_cons in new_cons:
            created_cons.append(added_cons)

    return created_vars, created_cons


def aggregate_estimator_outputs(_output, lr, constant, aggr="sum"):
    """
    Aggregate the output of individual estimators into a single expression for each output dimension.
    This function is needed for models with multiple estimators, e.g. gradient boosting decision trees and
    random forests.
    The output after aggregation can then be used as input for argmax classification.

    Parameters
    ----------
    _output : np.ndarray
        The output variables from each individual estimator (e.g. decision tree)
    lr : float
        The learning rate used for training and which is used to scale the output
    constant : np.ndarray
        The constant term that is added to the aggregation
    aggr : "sum" or "avg"
        Aggregation type ("sum" or "avg"). "Sum" for gradient boosting decision trees. "avg" for random forests.
    Returns
    -------

    aggregated_output : np.ndarray
        The new aggregated output per dimension over all estimators. Traditionally a sum over one dimension.

    """
    assert aggr in [
        "sum",
        "avg",
    ], f"Aggregation type {aggr} is neither sum or avg. No model exists."
    assert (
        _output.ndim == 3
    ), f"Aggregating estimator outputs of invalid dimension. {_output.ndim} != 3"

    n_samples = _output.shape[0]
    outdim = _output.shape[-1]
    n_estimators = _output.shape[1]

    aggregated_output = np.zeros((n_samples, outdim), dtype=object)
    for i in range(n_samples):
        for j in range(outdim):
            sum_expr = constant[j]
            for k in range(n_estimators):
                scale = 1 if aggr == "sum" else n_estimators
                sum_expr += lr * _output[i][k][j] / scale
            aggregated_output[i][j] = sum_expr

    return aggregated_output


def create_tree_estimators(
    scip_model,
    _input,
    tree_vars,
    trees,
    n_estimators,
    outdim,
    unique_naming_prefix,
    epsilon,
    classification,
    **kwargs,
):
    """
    Creates individual tree estimator models for each decision tree.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    _input : np.ndarray
        The input variables that are passed to each decision tree
    tree_vars : np.ndarray
        The PySCIPOpt variables that have been created to represent the output of each decision tree (i.e. estimator)
    trees : list
        A list of lists containing dictionary information that completely describe each decision tree (i.e. estimator)
    n_estimators : int
        The number of decision trees (i.e. estimators)
    outdim : int
        The output dimension of each decision tree
    unique_naming_prefix : str
        The unique naming prefix string that goes before all variables and constraints that are constructed by SCIP
    epsilon : float
        The epsilon that is used for each decision tree model. See #TODO: Decision tree modelling path
    classification : bool
        Whether the individual decision trees (i.e. estimators) are classification trees

    Returns
    -------

    estimators : list
        A list of :py:class`pyscipopt_ml.modelling.aggregate_tree_model.TreeEstimator`
    """

    estimators = []
    for i in range(n_estimators):
        for j in range(outdim):
            unique_prefix = unique_naming_prefix + f"{i}_{j}"
            estimators.append(
                TreeEstimator(
                    scip_model,
                    trees[i][j],
                    _input,
                    tree_vars[:, i, j].reshape((-1, 1)),
                    unique_prefix,
                    epsilon,
                    classification,
                    **kwargs,
                )
            )

    return estimators


def create_sklearn_tree_estimators(
    scip_model,
    predictor,
    _input,
    n_samples,
    outdim,
    unique_naming_prefix,
    classification,
    gbdt_or_rf="gbdt",
    **kwargs,
):
    """
    Create individual estimators for each decision tree for decision tree based ensemble predictors from SKLearn.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    predictor : GradientBoostingClassifier | GradientBoostingRegressor | RandomForestClassifier | RandomForestRegressor
        The Sklearn predictor that we are modelling
    _input : np.ndarray
        The input variables into each decision tree (i.e. estimator)
    n_samples : int
        The number of samples as input
    outdim : int
        The number of outputs of each estimator
    unique_naming_prefix : str
        The unique naming prefix string that goes before all variables and constraints that are constructed by SCIP
    classification : bool
        Whether the individual decision trees (i.e. estimators) are classification trees
    gbdt_or_rf : "gbdt" | "rf"
        Whether the predictor is for gradient boosting decision trees or random forests.

    Returns
    -------
    estimators : list
        A list of :py:class`pyscipopt_ml.modelling.aggregate_tree_model.TreeEstimator`
    tree_vars : np.ndarray
        A np.ndarray of created PySCIPopt vars
    """

    # Create variables to represent the output of each decision tree (i.e. estimator)
    shape = (n_samples, predictor.n_estimators, outdim)
    tree_vars = create_vars(
        scip_model, shape=shape, vtype="C", lb=None, name_prefix=unique_naming_prefix + "tree_var"
    )

    # Create each estimator. In the case of GBDT, there are (n_estimators, outdim) many estimators, while for RF
    # there are (outdim,) many estimators. In the case of GBDT for classification each individual DT is regression.
    estimators = []
    if gbdt_or_rf == "gbdt":
        for i in range(predictor.n_estimators_):
            for j in range(outdim):
                unique_prefix = unique_naming_prefix + f"{i}_{j}"
                tree = predictor.estimators_[i][j]
                estimators.append(
                    add_decision_tree_regressor_constr(
                        scip_model,
                        tree,
                        _input,
                        tree_vars[:, i, j].reshape((-1, 1)),
                        unique_prefix,
                        **kwargs,
                    )
                )
    elif gbdt_or_rf == "rf":
        for i in range(predictor.n_estimators):
            tree = predictor.estimators_[i]
            unique_prefix = unique_naming_prefix + f"{i}"
            if classification:
                estimators.append(
                    add_decision_tree_classifier_constr(
                        scip_model, tree, _input, tree_vars[:, i, :], unique_prefix, **kwargs
                    )
                )
            else:
                estimators.append(
                    add_decision_tree_regressor_constr(
                        scip_model, tree, _input, tree_vars[:, i, :], unique_prefix, **kwargs
                    )
                )

    return estimators, tree_vars


class TreeEstimator(AbstractPredictorConstr):
    def __init__(
        self,
        scip_model,
        tree,
        input_vars,
        output_vars,
        unique_naming_prefix,
        epsilon,
        classification,
        **kwargs,
    ):
        self._tree = tree
        self._epsilon = epsilon
        self._classification = classification
        AbstractPredictorConstr.__init__(
            self, scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        new_vars, new_cons = leaf_formulation(
            self.scip_model,
            self.input,
            self.output,
            self._tree,
            self.unique_naming_prefix,
            self._epsilon,
            classification=self._classification,
        )

        for new_var in new_vars:
            self._created_vars.append(new_var)
        for new_con in new_cons:
            self._created_cons.append(new_con)

    def get_error(self, eps):
        raise AssertionError("Tree estimator has no get_error functionality")
