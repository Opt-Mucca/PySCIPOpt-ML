"""Implements some utility tools for all scikit-learn objects."""

import numpy as np
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted

from ..exceptions import NoSolution
from ..modelling import AbstractPredictorConstr


class SKgetter(AbstractPredictorConstr):
    """Utility class for sklearn regression models convertors.

    Implement some common functionalities: check predictor is fitted, output dimension, get error

    Attributes
    ----------
    predictor
        Scikit-Learn predictor embedded into SCIP model.
    """

    def __init__(self, predictor, input_vars, output_type="regular", **kwargs):
        check_is_fitted(predictor)
        self.predictor = predictor
        # predictor._check_feature_names(input_vars, reset=False)
        # self.output_type = output_type
        # if hasattr(predictor, "n_features_in_"):
        #     self._input_shape = predictor.n_features_in_
        # if hasattr(predictor, "n_outputs_"):
        #     self._output_shape = predictor.n_outputs_

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
                sk_output_values = self.predictor.predict(self.input_values).reshape(
                    self.input.shape[0], self.output.shape[-1]
                )
            else:
                sk_class_prediction = self.predictor.predict(self.input_values)
                sk_output_values = np.zeros((self.input.shape[0], self.output.shape[-1]))
                for i, class_pred in enumerate(sk_class_prediction):
                    if self.output.shape[-1] == 1:
                        sk_output_values[i][0] = class_pred
                    else:
                        sk_output_values[i][class_pred] = 1
            scip_output_values = self.output_values
            error = np.abs(sk_output_values - scip_output_values)
            max_error = np.max(error)
            if eps is not None and max_error > eps:
                print(
                    f"SCIP output values of ML model {self.predictor} have larger than max error {max_error} > {eps}"
                )
            return error

        raise NoSolution()
