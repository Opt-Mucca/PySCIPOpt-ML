"""Module for formulating a
:external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor` or a
:external+sklearn:py:class:`sklearn.tree.DecisionTreeClassifier`
in a PySCIPOpt Model.
"""


from ..modelling import AbstractPredictorConstr
from ..modelling.decision_tree.decision_tree_model import leaf_formulation
from .skgetter import SKgetter


def add_decision_tree_regressor_constr(
    scip_model,
    decision_tree_regressor,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    epsilon=0.0,
    **kwargs,
):
    """Formulate decision_tree_regressor into a SCIP Model.

    The formulation predicts the values of output_vars using input_vars
    according to decision_tree_regressor.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    decision_tree_regressor : :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor`
        The decision tree regressor to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for decision tree in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for decision tree in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.
    Returns
    -------
    DecisionTreeRegressorConstr
        Object containing information about what was added to scip_model to formulate decision_tree_regressor

    Note
    ----

    |VariablesDimensionsWarn|
    """
    return DecisionTreeConstr(
        scip_model,
        decision_tree_regressor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        epsilon,
        False,
        **kwargs,
    )


def add_decision_tree_classifier_constr(
    scip_model,
    decision_tree_classifier,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    epsilon=0.0,
    **kwargs,
):
    """Formulate decision_tree_classifier into a SCIP Model.

    The formulation predicts the values of output_vars using input_vars
    according to decision_tree_classifier.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    decision_tree_classifier : :external+sklearn:py:class:`sklearn.tree.DecisionTreeClassifier`
        The decision tree classifier to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for decision tree in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for decision tree in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.
    Returns
    -------
    DecisionTreeClassifierConstr
        Object containing information about what was added to scip_model to formulate decision_tree_classifier

    Note
    ----

    |VariablesDimensionsWarn|

    Warning
    -------

    Although decision trees with multiple outputs are tested they were never
    used in a non-trivial optimization model. It should be used with care at
    this point.
    """
    return DecisionTreeConstr(
        scip_model,
        decision_tree_classifier,
        input_vars,
        output_vars,
        unique_naming_prefix,
        epsilon,
        True,
        **kwargs,
    )


class DecisionTreeConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor`
    or trained :external+sklearn:py:class:`sklearn.tree.DecisionTreeClassifier` with pyscipopt.

    |ClassShort|.
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars=None,
        unique_naming_prefix="",
        epsilon=0.0,
        classification=False,
        formulation="leafs",
        **kwargs,
    ):
        self.epsilon = epsilon
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
        tree = self.predictor.tree_

        tree_dict = {
            "children_left": tree.children_left,
            "children_right": tree.children_right,
            "feature": tree.feature,
            "threshold": tree.threshold,
            "value": tree.value,
            "node_count": tree.node_count,
            "n_features": tree.n_features,
        }

        new_vars, new_cons = leaf_formulation(
            self.scip_model,
            self.input,
            self.output,
            tree_dict,
            self.unique_naming_prefix,
            self.epsilon,
            classification=self.classification,
        )

        for new_var in new_vars:
            self._created_vars.append(new_var)
        for new_con in new_cons:
            self._created_cons.append(new_con)
