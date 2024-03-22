"""Module for formulating a :external+sklearn:py:class:`sklearn.preprocessing.StandardScalar`,
:external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures`,
:external+sklearn:py:class:`sklearn.preprocessing.Normalizer`, or
:external+sklearn:py:class:`sklearn.preprocessing.Binarizer`
into a PySCIPOpt Model.
"""

import numpy as np
from pyscipopt import sqrt

from ..exceptions import NoModel
from ..modelling.classification import max_formulation
from ..modelling.var_utils import create_vars
from .skgetter import SKtransformer


def add_binarizer_constr(
    scip_model,
    binarizer,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate binarizer into scip_model.

    Note that this function creates the output variables from
    the input variables.

    Parameters
    ----------
    scip_model : SCIP Model
        The PySCIPOpt Model where the polynomial features should be inserted.
    binarizer : :external+sklearn:py:class:`sklearn.preprocessing.Binarizer`
        The normalizer to insert into scip_model.
    input_vars : list or np.ndarray
        Decision variables used as input for the transformer in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for the transformer in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    sklearn.preprocessing.BinarizerConstr
        Object containing information about what was added to scip_model to insert the
        binarizer in it
    """
    return BinarizerConstr(
        scip_model, binarizer, input_vars, output_vars, unique_naming_prefix, **kwargs
    )


def add_normalizer_constr(
    scip_model,
    normalizer,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate normalizer into scip_model.

    Note that this function creates the output variables from
    the input variables.

    Parameters
    ----------
    scip_model : SCIP Model
        The PySCIPOpt Model where the polynomial features should be inserted.
    normalizer : :external+sklearn:py:class:`sklearn.preprocessing.Normalizer`
        The normalizer to insert into scip_model.
    input_vars : list or np.ndarray
        Decision variables used as input for the transformer in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for the transformer in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    sklearn.preprocessing.NormalizerConstr
        Object containing information about what was added to scip_model to insert the
        normalizer in it
    """
    return NormalizerConstr(
        scip_model, normalizer, input_vars, output_vars, unique_naming_prefix, **kwargs
    )


def add_polynomial_features_constr(
    scip_model,
    polynomial_features,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate polynomial_features into scip_model.

    Note that this function creates the output variables from
    the input variables.

    Parameters
    ----------
    scip_model : SCIP Model
        The PySCIPOpt Model where the polynomial features should be inserted.
    polynomial_features : :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures`
        The polynomial features to insert into scip_model.
    input_vars : list or np.ndarray
        Decision variables used as input for the transformer in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for the transformer in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    sklearn.preprocessing.PolynomialFeaturesConstr
        Object containing information about what was added to scip_model to insert the
        polynomial_features in it
    """
    return PolynomialFeaturesConstr(
        scip_model, polynomial_features, input_vars, output_vars, unique_naming_prefix, **kwargs
    )


def add_standard_scaler_constr(
    scip_model, standard_scaler, input_vars, output_vars=None, unique_naming_prefix="", **kwargs
):
    """Formulate standard_scaler into scip_model.

    Note that this function creates the output variables from
    the input variables.

    Parameters
    ----------
    scip_model : SCIP Model
        The PySCIPOpt Model where the polynomial features should be inserted.
    standard_scaler : :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler`
        The standard scalar to insert into scip_model.
    input_vars : list or np.ndarray
        Decision variables used as input for the transformer in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for the transformer in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    sklearn.preprocessing.StandardScalerConstr
        Object containing information about what was added to scip_model to insert the
        standard_scaler in it
    """
    return StandardScalerConstr(
        scip_model, standard_scaler, input_vars, output_vars, unique_naming_prefix, **kwargs
    )


class StandardScalerConstr(SKtransformer):
    """Class to formulate a fitted
    :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler` in a
    PySCIPOpt Model.
    """

    def __init__(
        self, scip_model, scaler, input_vars, output_vars, unique_naming_prefix, **kwargs
    ):
        self.output_size = scaler.n_features_in_
        super().__init__(
            scip_model, scaler, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""

        scale = self.transformer.scale_
        mean = self.transformer.mean_

        scaler_cons = np.zeros(self.input.shape, dtype=object)

        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[-1]):
                name = self.unique_naming_prefix + f"_scaler_{i}_{j}"
                scaler_cons[i][j] = self.scip_model.addCons(
                    self.input[i][j] - (scale[j] * self.output[i][j]) == mean[j], name=name
                )

        self._created_cons.append(scaler_cons)


class PolynomialFeaturesConstr(SKtransformer):
    """Class to formulate a fitted
    :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures` in a
    PySCIPOpt Model.
    """

    def __init__(
        self,
        scip_model,
        polynomial_features,
        input_vars,
        output_vars,
        unique_naming_prefix,
        **kwargs,
    ):
        self.output_size = polynomial_features.n_output_features_
        super().__init__(
            scip_model,
            polynomial_features,
            input_vars,
            output_vars,
            unique_naming_prefix,
            **kwargs,
        )

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""

        n_samples, n_features = self.input.shape
        powers = self.transformer.powers_
        assert powers.shape[0] == self.transformer.n_output_features_
        assert powers.shape[1] == n_features

        poly_cons = np.zeros((n_samples, self.transformer.n_output_features_), dtype=object)

        for i in range(n_samples):
            for j, power in enumerate(powers):
                scip_expr = 1.0
                for k, input_var in enumerate(self.input[i, :]):
                    if power[k] >= 1:
                        scip_expr *= input_var ** power[k]
                name = self.unique_naming_prefix + f"_poly_{i}_{j}"
                poly_cons[i][j] = self.scip_model.addCons(
                    self.output[i][j] == scip_expr, name=name
                )

        self._created_cons.append(poly_cons)


class NormalizerConstr(SKtransformer):
    """Class to formulate a fitted
    :external+sklearn:py:class:`sklearn.preprocessing.Normalizer` in a
    PySCIPOpt Model.
    """

    def __init__(
        self, scip_model, normalizer, input_vars, output_vars, unique_naming_prefix, **kwargs
    ):
        self.output_size = normalizer.n_features_in_
        if normalizer.norm not in ["l1", "l2", "max"]:
            raise NoModel(normalizer, f"Norm {normalizer.norm} is not supported.")
        super().__init__(
            scip_model, normalizer, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""

        norm = self.transformer.norm
        n_samples = self.input.shape[0]
        n_features = self.input.shape[-1]

        # Determine if slack variables need to be created
        pos_vars = True
        if norm in ["l1", "max"]:
            for i in range(n_samples):
                for j in range(n_features):
                    if self.input[i][j].getLbGlobal() < 0:
                        pos_vars = False
                        break
        if not pos_vars:
            pos_slack_vars = create_vars(
                self.scip_model,
                self.input.shape,
                vtype="C",
                lb=0,
                name_prefix=self.unique_naming_prefix + "_pos",
            )
            neg_slack_vars = create_vars(
                self.scip_model,
                self.input.shape,
                vtype="C",
                lb=0,
                name_prefix=self.unique_naming_prefix + "_neg",
            )
            dist_sum_cons = np.zeros(self.input.shape, dtype=object)
            for i in range(n_samples):
                for j in range(n_features):
                    name = f"_dist_slack_{i}_{j}"
                    dist_sum_cons[i][j] = self.scip_model.addCons(
                        self.input[i][j] == pos_slack_vars[i][j] - neg_slack_vars[i][j],
                        name=name,
                    )
            self._created_vars.append(pos_slack_vars)
            self._created_vars.append(neg_slack_vars)
            self._created_cons.append(dist_sum_cons)
        # If using max norm then need to get the max
        if norm == "max":
            max_scale_vars = create_vars(
                self.scip_model,
                (n_samples, 1),
                vtype="C",
                lb=None,
                name_prefix=self.unique_naming_prefix + "_max",
            )
            input_vars = self.input if pos_vars else pos_slack_vars + neg_slack_vars
            max_form_vars, max_form_cons = max_formulation(
                self.scip_model,
                input_vars,
                max_scale_vars,
                self.unique_naming_prefix,
            )
            for scip_var in max_form_vars:
                self._created_vars.append(scip_var)
            for scip_cons in max_form_cons:
                self._created_cons.append(scip_cons)

        # Create the normalisation constraints
        norm_cons = np.zeros(self.input.shape, dtype=object)
        for i in range(n_samples):
            if norm == "l2":
                scale = sqrt(sum(self.input[i][j] ** 2 for j in range(n_features)))
            elif norm == "l1":
                if pos_vars:
                    scale = sum(self.input[i][j] for j in range(n_features))
                else:
                    scale = sum(
                        pos_slack_vars[i][j] + neg_slack_vars[i][j] for j in range(n_features)
                    )
            elif norm == "max":
                scale = max_scale_vars[i][0]
            else:
                raise NoModel(self.transformer, f"Norm {norm} is not supported.")
            for j in range(n_features):
                name = f"norm_cons_{i}_{j}"
                norm_cons[i][j] = self.scip_model.addCons(
                    self.output[i][j] == self.input[i][j] / scale, name=name
                )

        self._created_cons.append(norm_cons)


class BinarizerConstr(SKtransformer):
    """Class to formulate a fitted
    :external+sklearn:py:class:`sklearn.preprocessing.Binarizer` in a
    PySCIPOpt Model.
    """

    def __init__(
        self, scip_model, binarizer, input_vars, output_vars, unique_naming_prefix, **kwargs
    ):
        self.output_size = binarizer.n_features_in_
        self.classification = True
        super().__init__(
            scip_model, binarizer, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""

        n_samples = self.input.shape[0]
        n_features = self.input.shape[-1]
        threshold = self.transformer.threshold

        binary_indicator_cons = np.zeros((n_samples, n_features, 2), dtype=object)

        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[-1]):
                name = self.unique_naming_prefix + f"_binary_ind_{i}_{j}_0"
                binary_indicator_cons[i][j][0] = self.scip_model.addConsIndicator(
                    self.input[i][j] <= threshold, self.output[i][j], False, name=name
                )
                name = self.unique_naming_prefix + f"_binary_ind_{i}_{j}_1"
                binary_indicator_cons[i][j][1] = self.scip_model.addConsIndicator(
                    -self.input[i][j] <= -threshold, self.output[i][j], True, name=name
                )

        self._created_cons.append(binary_indicator_cons)


def sklearn_transformers():
    """Return dictionary of Scikit Learn preprocessing objects."""
    return {
        "StandardScaler": add_standard_scaler_constr,
        "PolynomialFeatures": add_polynomial_features_constr,
        "Normalizer": add_normalizer_constr,
        "Binarizer": add_binarizer_constr,
    }
