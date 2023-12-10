"""Module for formulating a XGBoost gradient boosting or random forest regressor / classifier
 into a PySCIPOpt Model. """

import numpy as np

from ..modelling import AbstractPredictorConstr
from ..modelling.gradient_boosting import aggregated_estimator_formulation
from .xgbgetter import XGBgetter


def add_xgbregressor_constr(
    scip_model,
    xgboost_regressor,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    epsilon=0.0,
    **kwargs
):
    """Formulate xgboost_regressor (gradient boosting decision tree) as constraints into a pyscipopt Model.

    The formulation predicts the values of output_vars using input_vars
    according to xgboost_regressor.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The pyscipopt Model where the predictor should be inserted.
    xgboost_regressor : :external+xgb:py:class:`xgboost.XGBRegressor`
        The gradient boosting regressor to insert as predictor.
    input_vars : list or dict
        Decision variables used as input for gradient boosting regressor in model.
    output_vars : list or dict, optional
        Decision variables used as output for gradient boosting regressor in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.

    Returns
    -------
    XGBoostRegressorConstr
        Object containing information about what was added to scip_model to formulate
        gradient_boosting_regressor.

    Note
    ----
    |VariablesDimensionsWarn|

    Raises
    ------
    NoModel
        If the booster is not of type "gbtree".
    """
    return XGBoostConstr(
        scip_model,
        xgboost_regressor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        epsilon=epsilon,
        aggr="sum",
        classification=False,
        **kwargs,
    )


def add_xgbclassifier_constr(
    scip_model,
    xgboost_classifier,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    epsilon=0.0,
    **kwargs
):
    """Formulate xgboost_classifier (gradient boosting decision tree) as constraints into a pyscipopt Model.

    The formulation predicts the values of output_vars using input_vars
    according to xgboost_classifier.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The pyscipopt Model where the predictor should be inserted.
    xgboost_classifier : :external+xgb:py:class:`xgboost.XGBClassifier`
        The gradient boosting classifier to insert as predictor.
    input_vars : list or dict
        Decision variables used as input for gradient boosting regressor in model.
    output_vars : list or dict, optional
        Decision variables used as output for gradient boosting regressor in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.

    Returns
    -------
    XGBoostClassifierConstr
        Object containing information about what was added to scip_model to formulate
        gradient_boosting_classifier.

    Note
    ----
    |VariablesDimensionsWarn|

    Raises
    ------
    NoModel
        If the booster is not of type "gbtree".
    """

    return XGBoostConstr(
        scip_model,
        xgboost_classifier,
        input_vars,
        output_vars,
        unique_naming_prefix,
        epsilon=epsilon,
        aggr="sum",
        classification=True,
        **kwargs,
    )


def add_xgbregressor_rf_constr(
    scip_model,
    xgboost_regressor,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    epsilon=0.0,
    **kwargs
):
    """Formulate xgboost_regressor (random forest) as constraints into a pyscipopt Model.

    The formulation predicts the values of output_vars using input_vars
    according to xgboost_regressor.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The pyscipopt Model where the predictor should be inserted.
    xgboost_regressor : :external+xgb:py:class:`xgboost.XGBRFRegressor`
        The gradient boosting regressor to insert as predictor.
    input_vars : list or dict
        Decision variables used as input for gradient boosting regressor in model.
    output_vars : list or dict, optional
        Decision variables used as output for gradient boosting regressor in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.

    Returns
    -------
    XGBoostRegressorConstr
        Object containing information about what was added to scip_model to formulate
        gradient_boosting_regressor.

    Note
    ----
    |VariablesDimensionsWarn|

    Raises
    ------
    NoModel
        If the booster is not of type "gbtree".
    """
    return XGBoostConstr(
        scip_model,
        xgboost_regressor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        epsilon=epsilon,
        aggr="sum",
        classification=False,
        **kwargs,
    )


def add_xgbclassifier_rf_constr(
    scip_model,
    xgboost_classifier,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    epsilon=0.0,
    **kwargs
):
    """Formulate xgboost_classifier (random forest) as constraints into a pyscipopt Model.

    The formulation predicts the values of output_vars using input_vars
    according to xgboost_classifier.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The pyscipopt Model where the predictor should be inserted.
    xgboost_classifier : :external+xgb:py:class:`xgboost.XGBClassifier`
        The gradient boosting classifier to insert as predictor.
    input_vars : list or dict
        Decision variables used as input for gradient boosting regressor in model.
    output_vars : list or dict, optional
        Decision variables used as output for gradient boosting regressor in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.

    Returns
    -------
    XGBoostClassifierConstr
        Object containing information about what was added to scip_model to formulate
        gradient_boosting_classifier.

    Note
    ----
    |VariablesDimensionsWarn|

    Raises
    ------
    NoModel
        If the booster is not of type "gbtree".
    """

    return XGBoostConstr(
        scip_model,
        xgboost_classifier,
        input_vars,
        output_vars,
        unique_naming_prefix,
        epsilon=epsilon,
        aggr="sum",
        classification=True,
        **kwargs,
    )


class XGBoostConstr(XGBgetter, AbstractPredictorConstr):
    """Class to model trained :external+xgb:py:class:`xgboost.XGBRegressor` or
    :external+xgb:py:class:`xgboost.XGBClassifier` with constraints in a pyscipopt Model.

    |ClassShort|
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars,
        unique_naming_prefix="",
        epsilon=0.0,
        aggr="sum",
        classification=False,
        **kwargs
    ):
        XGBgetter.__init__(self, predictor, input_vars, **kwargs)
        self.estimators_ = []
        self.classification = classification
        self.epsilon = epsilon
        self.aggr = aggr
        # Get the output dimension
        if hasattr(predictor, "predict_proba"):
            dummy_prediction = predictor.predict_proba(np.zeros((1, predictor.n_features_in_)))
            output_size = dummy_prediction.shape[-1]
            if self.classification and output_size <= 2:
                self.output_size = 1
            else:
                self.output_size = output_size
        else:
            dummy_prediction = predictor.predict(np.zeros((1, predictor.n_features_in_)))
            self.output_size = dummy_prediction.shape[-1]
        AbstractPredictorConstr.__init__(
            self, scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):

        # Extract the information on the trained model and create new variables
        trees, constant, tree_vars = self.extract_raw_data_and_create_tree_vars(
            epsilon=self.epsilon
        )
        self._created_vars.append(tree_vars)

        # Construct each individual tree estimator and the constraints that link them
        constant = np.array([constant for _ in range(self.output.shape[-1])])
        estimators, created_vars, created_cons = aggregated_estimator_formulation(
            self.scip_model,
            self.input,
            self.output,
            tree_vars,
            trees,
            constant,
            1,
            self.predictor.n_estimators,
            self.unique_naming_prefix,
            self.epsilon,
            self.aggr,
            self.classification,
            **kwargs,
        )

        for estimator in estimators:
            self.estimators_.append(estimator)
        for created_var in created_vars:
            self._created_vars.append(created_var)
        for created_con in created_cons:
            self._created_cons.append(created_con)
