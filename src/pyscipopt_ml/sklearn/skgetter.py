"""Implements some utility tools for all scikit-learn objects."""

import numpy as np
from sklearn.base import ClusterMixin, is_classifier
from sklearn.multioutput import _MultiOutputEstimator
from sklearn.utils.validation import check_is_fitted

from ..exceptions import NoSolution
from ..modelling import AbstractPredictorConstr


class SKgetter(AbstractPredictorConstr):
    """Utility class for sklearn models convertors.

    Implement some common functionalities: check predictor is fitted, get error

    Attributes
    ----------
    predictor
        Scikit-Learn predictor embedded into SCIP model.
    """

    def __init__(self, predictor, **kwargs):
        check_is_fitted(predictor)
        self.predictor = predictor

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
            if (
                not is_classifier(self.predictor) and not isinstance(self.predictor, ClusterMixin)
            ) or isinstance(self.predictor, _MultiOutputEstimator):
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


class SKtransformer(AbstractPredictorConstr):
    """Utility class for sklearn preprocessing models convertors.

    Implement some common functionalities.

    Attributes
    ----------
    transformer
        Scikit-Learn transformer embedded into SCIP Model.
    """

    def __init__(
        self,
        scip_model,
        transformer,
        input_vars,
        output_vars=None,
        unique_naming_prefix="",
        **kwargs,
    ):
        self.transformer = transformer
        check_is_fitted(transformer)
        super().__init__(scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs)
        # As transforming units can only ever be used as non-final stages of a pipeline, the output are created vars
        self._created_vars.append(self.output)

    def get_error(self, eps=None):
        """Returns error in SCIP's solution with respect to the actual output of the trained predictor

        Parameters
        ----------
        eps : float or int or None, optional
            The maximum allowed tolerance for a mismatch between the actual predictive model and SCIP.
            If the error is larger than eps an appropriate warning is printed

        Returns
        -------
        error: np.ndarray
            The absolute values of the difference between SCIP's solution and the trained ML model's output given
            the input as defined by SCIP. The matrix is the same dimension as the output of the fitted transformer.
            Using sklearn / pyscipopt, the absolute difference between
            transformer.transform(input) and scip.getVal(output).

        Raises
        ------
        NoSolution
            If SCIP has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution:
            transformer = self.transformer
            input_values = self.input_values

            transformed = transformer.transform(input_values)
            if len(transformed.shape) == 1:
                transformed = transformed.reshape(-1, 1)

            r_val = np.abs(transformed - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{transformed} != {self.output_values}")
            return r_val

        raise NoSolution()
