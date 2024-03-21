"""Module for formulating a :external+sklearn:py:class:`sklearn.svm.SVR`,
:external+sklearn:py:class:`sklearn.svm.SVC`,
:external+sklearn:py:class:`sklearn.svm.LinearSVR`, or
:external+sklearn:py:class:`sklearn.svm.LinearSVC` into a PySCIPOpt Model.
"""


import numpy as np
from sklearn.svm import LinearSVC, LinearSVR

from ..exceptions import NoModel
from ..modelling import AbstractPredictorConstr
from ..modelling.classification import argmax_bound_formulation
from ..modelling.var_utils import create_vars
from .skgetter import SKgetter


def add_support_vector_regressor_constr(
    scip_model,
    support_vector_regressor,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate support_vector_regressor in scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    support_vector_regressor.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the predictor should be inserted.
    support_vector_regressor : :external+sklearn:py:class:`sklearn.svm.SVR`
        The support vector regressor to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for support vector in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for support vector in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    SupportVectorConstr
       Object containing information about what was added to scip_model to formulate
       support_vector_regressor.

    Note
    ----
    |VariablesDimensionsWarn|
    """

    return SupportVectorConstr(
        scip_model,
        support_vector_regressor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        False,
        **kwargs,
    )


def add_support_vector_classifier_constr(
    scip_model,
    support_vector_classifier,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate support_vector_classifier in scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    support_vector_classifier.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the predictor should be inserted.
    support_vector_classifier : :external+sklearn:py:class:`sklearn.svm.SVC`
        The support vector classifier to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for support vector in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for support vector in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    SupportVectorConstr
       Object containing information about what was added to scip_model to formulate
       support_vector_classifier.

    Note
    ----
    |VariablesDimensionsWarn|
    """

    return SupportVectorConstr(
        scip_model,
        support_vector_classifier,
        input_vars,
        output_vars,
        unique_naming_prefix,
        True,
        **kwargs,
    )


class SupportVectorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained :external+sklearn:py:class:`sklearn.svm.SVR`,
    :external+sklearn:py:class:`sklearn.svm.SVC`,
    :external+sklearn:py:class:`sklearn.svm.LinearSVR`, or
    :external+sklearn:py:class:`sklearn.svm.LinearSVC` with SCIP

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
        self.classification = classification
        if self.classification:
            if len(predictor.classes_) <= 2:
                self.output_size = 1
            else:
                raise NoModel(predictor, "Multi-classification is not supported for SVC")
        else:
            self.output_size = 1
        if isinstance(predictor, LinearSVR) or isinstance(predictor, LinearSVC):
            self.kernel = "linear"
        else:
            self.kernel = predictor.kernel
            if self.kernel not in ["linear", "poly"]:
                raise NoModel(predictor, f"Kernel type {self.kernel} not linear nor poly")
            if predictor.class_weight is not None:
                raise NoModel(predictor, "Non uniform class weights are not supported.")
        SKgetter.__init__(self, predictor, **kwargs)
        AbstractPredictorConstr.__init__(
            self, scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Predict output variables y from input variables X using the support vectors.

        Both X and y should be arrays or lists of variables of conforming dimensions.
        """

        n_samples = self.input.shape[0]
        n_features = self.input.shape[-1]

        # Extract data from the predictor object
        intercept = self.predictor.intercept_[0]
        if self.kernel == "linear":
            # LinearSVM has a different dimension output to SVM
            if self.predictor.coef_.ndim > 1:
                coefs = self.predictor.coef_[0]
            else:
                coefs = self.predictor.coef_
        else:
            degree = self.predictor.degree
            n_support = np.sum(self.predictor.n_support_)
            dual_coefs = self.predictor.dual_coef_
            support_vectors = self.predictor.support_vectors_
            gamma = self.predictor._gamma  # TODO: Is this possible to access otherwise?

        # Create additional variables for classification
        if self.classification:
            regression_vars = create_vars(
                self.scip_model,
                shape=(n_samples, 1),
                vtype="C",
                lb=None,
                ub=None,
                name_prefix=self.unique_naming_prefix + "reg_",
            )

        # Create the regression constraints for the SVR / SVC
        regression_cons = np.zeros((n_samples,), dtype=object)
        for i in range(n_samples):
            if not self.classification:
                out_var = self.output[i][0]
            else:
                out_var = regression_vars[i][0]
            if self.kernel == "linear":
                name = self.unique_naming_prefix + f"reg_sv_lin_"
                regression_cons[i] = self.scip_model.addCons(
                    sum(self.input[i][j] * coefs[j] for j in range(n_features)) + intercept
                    == out_var,
                    name=name,
                )
            else:
                name = self.unique_naming_prefix + f"reg_sv_poly_{i}"
                dot_products = [
                    sum(self.input[i][j] * support_vectors[s][j] for j in range(n_features))
                    for s in range(n_support)
                ]
                regression_cons[i] = self.scip_model.addCons(
                    sum(
                        dual_coefs[0][s] * (gamma**degree) * dot_products[s] ** degree
                        for s in range(n_support)
                    )
                    + intercept
                    == out_var,
                    name=name,
                )

        # In the case of classification add conversion to binaries (centres at 0)
        if self.classification:
            argmax_vars, argmax_cons = argmax_bound_formulation(
                self.scip_model,
                regression_vars,
                self.output,
                self.unique_naming_prefix,
                one_dim_center=0,
            )
            self._created_vars.append(regression_vars)
            for new_vars in argmax_vars:
                self._created_vars.append(new_vars)
            for new_cons in argmax_cons:
                self._created_cons.append(new_cons)

        self._created_cons.append(regression_cons)
