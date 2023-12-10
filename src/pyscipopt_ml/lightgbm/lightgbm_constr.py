"""Module for formulating a LightGBM gradient boosting or random forest regressor / classifier into
a PySCIPOpt Model."""

import numpy as np

from ..modelling import AbstractPredictorConstr
from ..modelling.gradient_boosting import aggregated_estimator_formulation
from .lgbgetter import LGBgetter


def add_lgbregressor_constr(
    scip_model,
    lightgbm_regressor,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    epsilon=0.0,
    **kwargs
):
    """Formulate lightgbm_regressor as constraints into a pyscipopt Model. Accommodates both Gradient Boosting
    Decision Trees and Random Forests as boosting types.

    The formulation predicts the values of output_vars using input_vars
    according to lightgbm_regressor.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The pyscipopt Model where the predictor should be inserted.
    lightgbm_regressor : :external+lgb:py:class:`lightgbm.LGBMRegressor`
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
    LightGBMRegressorConstr
        Object containing information about what was added to scip_model to formulate
        gradient_boosting_regressor.

    Raises
    ------
    NoModel
        If the boosting type is not "gbdt" or "rf".

    Note
    ----
    |VariablesDimensionsWarn|
    """
    return LightGBMConstr(
        scip_model,
        lightgbm_regressor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        epsilon=epsilon,
        classification=False,
        **kwargs,
    )


def add_lgbclassifier_constr(
    scip_model,
    lightgbm_classifier,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    epsilon=0.0,
    **kwargs
):
    """Formulate lightgbm_classifier as constraints into a pyscipopt Model. Accommodates both Gradient Boosting
    Decision Trees and Random Forests as boosting types.

    The formulation predicts the values of output_vars using input_vars
    according to lightgbm_classifier.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The pyscipopt Model where the predictor should be inserted.
    lightgbm_classifier : :external+lgb:py:class:`lightgbm.LGBMClassifier`
        The gradient boosting classifier to insert as predictor.
    input_vars : list or dict
        Decision variables used as input for gradient boosting classifier in model.
    output_vars : list or dict, optional
        Decision variables used as output for gradient boosting classifier in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.

    Returns
    -------
    LightGBMClassifierConstr
        Object containing information about what was added to scip_model to formulate
        gradient_boosting_classifier.

    Raises
    ------
    NoModel
        If the boosting type is not "gbdt" or "rf".

    Note
    ----
    |VariablesDimensionsWarn|
    """
    return LightGBMConstr(
        scip_model,
        lightgbm_classifier,
        input_vars,
        output_vars,
        unique_naming_prefix,
        epsilon=epsilon,
        classification=True,
        **kwargs,
    )


class LightGBMConstr(LGBgetter, AbstractPredictorConstr):
    """Class to model trained :external+lgb:py:class:`lightgbm.LGBMRegressor` or
     :external+lgb:py:class:`lightgbm.LGBMClassifier` with constraints in a pyscipopt Model.

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
        classification=False,
        **kwargs
    ):
        LGBgetter.__init__(self, predictor, input_vars, **kwargs)
        self.estimators_ = []
        self.classification = classification
        self.epsilon = epsilon
        if self.classification:
            if predictor.n_classes_ <= 2:
                self.output_size = 1
            else:
                self.output_size = predictor.n_classes_
        else:
            # LGB only accommodates single output regression
            self.output_size = 1
        AbstractPredictorConstr.__init__(
            self, scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):

        # Extract the information on the trained model and create new variables
        trees, tree_vars = self.extract_raw_data_and_create_tree_vars(epsilon=self.epsilon)
        self._created_vars.append(tree_vars)

        # Construct each individual tree estimator and the constraints that link them
        constant = np.zeros((self.output.shape[-1],))
        aggr = "sum" if self.predictor.boosting_type == "gbdt" else "avg"
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
            aggr,
            self.classification,
            **kwargs,
        )

        for estimator in estimators:
            self.estimators_.append(estimator)
        for created_var in created_vars:
            self._created_vars.append(created_var)
        for created_con in created_cons:
            self._created_cons.append(created_con)
