"""Implements some utility tools for all lightgbm objects."""
import numpy as np
from sklearn.base import is_classifier

from ..exceptions import NoModel, NoSolution, ParameterError
from ..modelling import AbstractPredictorConstr
from ..modelling.var_utils import create_vars


class LGBgetter(AbstractPredictorConstr):
    """Utility class for lightgbm models convertors.

    Implement some common functionalities: check predictor is fitted, output dimension, get error

    Attributes
    ----------
    predictor
        Lightgbm predictor embedded into SCIP model.
    """

    def __init__(self, predictor, input_vars, output_type="regular", **kwargs):
        if not hasattr(predictor, "booster_"):
            raise ParameterError(
                "LightGBM model has not yet been fitted. There is nothing to model."
            )
        if predictor.boosting_type not in ["gbdt", "rf"]:
            raise NoModel(
                predictor,
                f"There is only support for LightGBM boosting type gbdt and rf. "
                f"Not {predictor.boosting_type}",
            )
        self.predictor = predictor

    def extract_raw_data_and_create_tree_vars(self, epsilon=0.0):
        """
        Function for extracting information from lgb._Booster and creating additional modelling variables.

        Parameters
        ----------
        epsilon : float, optional
            Small value used to impose strict inequalities for splitting nodes in
            MIP formulations.

        Returns
        -------
        trees : list
            A list of tree dictionaries similar in structure to that provided by SKlearn

        tree_vars : np.ndarray
            A numpy array filled with variables that represent output of trees from the trained LightGBM model
        """

        n_samples = self.input.shape[0]
        outdim = self.output.shape[1]

        # Extract the raw data
        raw = self.predictor.booster_.dump_model()
        n_features = self.predictor.n_features_
        n_trees = len(raw["tree_info"])
        n_estimators = self.predictor.n_estimators_

        def read_node(tree, tree_structure):
            # Increase the node count and keep track of the current node index
            node_idx = tree["node_count"]
            tree["node_count"] += 1

            # Add features specific to the node of the tree
            if "split_feature" in tree_structure:
                tree["feature"].append(tree_structure["split_feature"])
            else:
                tree["feature"].append(-1)
            if "threshold" in tree_structure:
                tree["threshold"].append(tree_structure["threshold"])
            else:
                tree["threshold"].append(-1)
            if "internal_value" in tree_structure:
                tree["value"].append([[tree_structure["internal_value"]]])
            else:
                tree["value"].append([[tree_structure["leaf_value"]]])
            if "decision_type" in tree_structure and tree_structure["decision_type"] != "<=":
                raise ParameterError(
                    f"Currently no support for LightGBM trees with decision type "
                    f"{tree_structure['decision_type']}"
                )

            # Add in dummy left and right children
            tree["children_left"].append(-1)
            tree["children_right"].append(-1)
            # Now recursively call a new node
            if "left_child" in tree_structure:
                tree["children_left"][node_idx] = tree["node_count"]
                tree = read_node(tree, tree_structure["left_child"])
            if "right_child" in tree_structure:
                tree["children_right"][node_idx] = tree["node_count"]
                tree = read_node(tree, tree_structure["right_child"])

            return tree

        trees = [
            [
                {
                    "node_count": 0,
                    "n_features": n_features,
                    "children_left": [],
                    "children_right": [],
                    "feature": [],
                    "threshold": [],
                    "value": [],
                }
                for _ in range(outdim)
            ]
            for _ in range(n_estimators)
        ]
        trees_converted = [0 for _ in range(outdim)]
        for i in range(n_trees):
            tree_raw = raw["tree_info"][i]["tree_structure"]
            class_idx = i % outdim
            tree_idx = trees_converted[class_idx]
            trees[tree_idx][class_idx] = read_node(trees[tree_idx][class_idx], tree_raw)
            trees[tree_idx][class_idx]["children_left"] = np.array(
                trees[tree_idx][class_idx]["children_left"], dtype=np.int32
            )
            trees[tree_idx][class_idx]["children_right"] = np.array(
                trees[tree_idx][class_idx]["children_right"], dtype=np.int32
            )
            trees[tree_idx][class_idx]["feature"] = np.array(
                trees[tree_idx][class_idx]["feature"], dtype=np.int32
            )
            trees[tree_idx][class_idx]["threshold"] = np.array(
                trees[tree_idx][class_idx]["threshold"], dtype=np.float64
            )
            trees[tree_idx][class_idx]["value"] = np.array(
                trees[tree_idx][class_idx]["value"], dtype=np.float64
            )
            trees_converted[class_idx] += 1

        shape = (n_samples, n_estimators, outdim)
        tree_vars = create_vars(
            self.scip_model, shape=shape, vtype="C", lb=None, ub=None, name_prefix="tree"
        )

        return trees, tree_vars

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
                lgb_output_values = self.predictor.predict(self.input_values).reshape(
                    self.input.shape[0], self.output.shape[-1]
                )
            else:
                lgb_class_prediction = self.predictor.predict(self.input_values)
                lgb_output_values = np.zeros((self.input.shape[0], self.output.shape[-1]))
                for i, class_pred in enumerate(lgb_class_prediction):
                    if self.output.shape[-1] == 1:
                        lgb_output_values[i][0] = class_pred
                    else:
                        lgb_output_values[i][class_pred] = 1
            scip_output_values = self.output_values
            error = np.abs(lgb_output_values - scip_output_values)
            max_error = np.max(error)
            if eps is not None and max_error > eps:
                print(
                    f"SCIP output values of ML model {self.predictor} have larger than max error {max_error} > {eps}"
                )
            return error

        raise NoSolution()
