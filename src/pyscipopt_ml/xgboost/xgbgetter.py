"""Implements some utility tools for all xgboost objects."""

import json

import numpy as np
from sklearn.base import is_classifier

from ..exceptions import NoModel, NoSolution, ParameterError
from ..modelling import AbstractPredictorConstr
from ..modelling.var_utils import create_vars


class XGBgetter(AbstractPredictorConstr):
    """Utility class for xgboost models convertors.

    Implement some common functionalities: check predictor is fitted, output dimension, get error

    Attributes
    ----------
    predictor
        Xgboost predictor embedded into SCIP model.
    """

    def __init__(self, predictor, input_vars, output_type="regular", **kwargs):
        if not hasattr(predictor, "_Booster"):
            raise ParameterError(
                "XGBoost model has not yet been fitted. There is nothing to model."
            )
        self.predictor = predictor

    def extract_raw_data_and_create_tree_vars(self, epsilon=0.0):
        """
        Function for extracting information from xgb.Booster and creating additional modelling variables.

        Parameters
        ----------
        epsilon : float, optional
            Small value used to impose strict inequalities for splitting nodes in
            MIP formulations.

        Returns
        -------
        trees : list
            A list of tree dictionaries similar in structure to that provided by SKlearn

        constant : float
            A constant value that is added to all regression output of the decision trees

        tree_vars : np.ndarray
            A numpy array filled with variables that represent output of trees from the trained XGBoost model
        """

        n_samples = self.input.shape[0]
        outdim = self.output.shape[1]

        # Extract the raw data
        xgb_raw = json.loads(self.predictor.get_booster().save_raw(raw_format="json"))
        model_params = xgb_raw["learner"]["learner_model_param"]
        n_trees = len(xgb_raw["learner"]["gradient_booster"]["model"]["trees"])
        constant = float(xgb_raw["learner"]["learner_model_param"]["base_score"])

        # Raise a warning if predictor cannot be modelled
        if int(model_params["num_class"]) > 1 and int(model_params["num_target"]) > 1:
            raise NoModel(self.predictor, "Multi-regression for multiple classes is not allowed")
        booster_type = xgb_raw["learner"]["gradient_booster"]["name"]
        if booster_type != "gbtree":
            raise NoModel(self.predictor, f"model not implemented for {booster_type}")
        assert (
            int(xgb_raw["learner"]["learner_model_param"]["num_target"]) == outdim
            or int(xgb_raw["learner"]["learner_model_param"]["num_class"]) == outdim
        ), "Dimension error when constructing XGBoost representation."
        assert (
            n_trees % outdim == 0
        ), "Uneven amount of estimator trees per class / output for XGBoost.Booster"

        # Convert the raw data into a format similar to sk-learn
        trees = [[{} for _ in range(outdim)] for _ in range(self.predictor.n_estimators)]
        trees_converted = [0 for _ in range(outdim)]
        for i in range(n_trees):
            tree_class = xgb_raw["learner"]["gradient_booster"]["model"]["tree_info"][i]
            tree_raw = xgb_raw["learner"]["gradient_booster"]["model"]["trees"][i]
            tree = trees[trees_converted[tree_class]][tree_class]
            tree["threshold"] = np.array(tree_raw["split_conditions"], dtype=np.float32)
            tree["children_left"] = np.array(tree_raw["left_children"])
            tree["children_right"] = np.array(tree_raw["right_children"])
            tree["feature"] = np.array(tree_raw["split_indices"])
            tree["node_count"] = len(tree_raw["split_conditions"])  # Also accessible via num_nodes
            tree["n_features"] = int(tree_raw["tree_param"]["num_feature"])
            tree["value"] = tree["threshold"].reshape((tree["node_count"], 1, 1))
            trees_converted[tree_class] += 1
        for i in range(outdim):
            if trees_converted[i] != trees_converted[0]:
                raise ParameterError("Uneven amount of estimators per output")

        shape = (n_samples, n_trees // outdim, outdim)
        tree_vars = create_vars(
            self.scip_model, shape=shape, vtype="C", lb=None, ub=None, name_prefix="tree"
        )

        return trees, constant, tree_vars

    def get_error(self, eps=None):
        """
        Returns error in SCIP's solution with respect to the actual output of the trained predictor

        Parameters
        ----------
        eps : float or int or None, optional
            The maximum allowed tolerance for a mismatch between the actual predictive model and SCIP.
            If the error is larger than eps an appropriate warning is printed

        Returns
        -------
        error: np.ndarray
            The absolute values of the difference between SCIP's solution and the trained ML model's output given
            the input as defined by SCIP. The matrix is the same dimension as the output of the trained predictor.
            Using sklearn / pyscipopt, the absolute difference between model.predict(input) and scip.getVal(output).

        Raises
        ------
        NoSolution
            If SCIP has no solution (either was not optimized or is infeasible).
        """

        if self._has_solution:
            if not is_classifier(self.predictor):
                xgb_output_values = self.predictor.predict(self.input_values).reshape(
                    self.input.shape[0], self.output.shape[-1]
                )
            else:
                xgb_class_prediction = self.predictor.predict(self.input_values)
                xgb_output_values = np.zeros((self.input.shape[0], self.output.shape[-1]))
                for i, class_pred in enumerate(xgb_class_prediction):
                    if self.output.shape[-1] == 1:
                        xgb_output_values[i][0] = class_pred
                    else:
                        xgb_output_values[i][class_pred] = 1
            scip_output_values = self.output_values
            error = np.abs(xgb_output_values - scip_output_values)
            max_error = np.max(error)
            if eps is not None and max_error > eps:
                print(
                    f"SCIP output values of ML model {self.predictor} have larger than max error {max_error} > {eps}"
                )
            return error

        raise NoSolution()
