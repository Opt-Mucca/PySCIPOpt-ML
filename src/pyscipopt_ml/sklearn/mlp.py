"""Module for formulating a :external+sklearn:py:class:`sklearn.neural_network.MLPRegressor` or
:external+sklearn:py:class:`sklearn.neural_network.MLPClassifier` in a PySCIPOpt Model."""
from ..exceptions import NoModel
from ..modelling.classification import argmax_bound_formulation
from ..modelling.neuralnet import BaseNNConstr
from ..modelling.var_utils import create_vars
from .skgetter import SKgetter


def add_mlp_regressor_constr(
    scip_model, mlp_regressor, input_vars, output_vars=None, unique_naming_prefix="", **kwargs
):
    """Formulate mlp_regressor into scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    mlp_regressor.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the predictor should be inserted.
    mlp_regressor : :external+sklearn:py:class:`sklearn.neural_network.MLPRegressor`
        The multi-layer perceptron regressor to insert as predictor.
    input_vars : np.ndarray or list
        Decision variables used as input for regression in model.
    output_vars : np.ndarray or list, optional
        Decision variables used as output for regression in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    MLPRegressorConstr
        Object containing information about what was added to scip_model to formulate
        mlp_regressor.

    Raises
    ------
    NoModel
        If the translation to SCIP of the activation function for the network is not
        implemented.

    Note
    ----
    |VariablesDimensionsWarn|
    """
    return MLPConstr(
        scip_model, mlp_regressor, input_vars, output_vars, unique_naming_prefix, False, **kwargs
    )


def add_mlp_classifier_constr(
    scip_model, mlp_classifier, input_vars, output_vars=None, unique_naming_prefix="", **kwargs
):
    """Formulate mlp_classifier into scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    mlp_classifier.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the predictor should be inserted.
    mlp_classifier : :external+sklearn:py:class:`sklearn.neural_network.MLPClassifier`
        The multi-layer perceptron classifier to insert as predictor.
    input_vars : np.ndarray or list
        Decision variables used as input for regression in model.
    output_vars : np.ndarray or list, optional
        Decision variables used as output for regression in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    MLPConstr
        Object containing information about what was added to scip_model to formulate
        mlp_classifier.

    Raises
    ------
    NoModel
        If the translation to SCIP of the activation function for the network is not
        implemented.

    Note
    ----
    |VariablesDimensionsWarn|
    """
    return MLPConstr(
        scip_model, mlp_classifier, input_vars, output_vars, unique_naming_prefix, True, **kwargs
    )


class MLPConstr(SKgetter, BaseNNConstr):
    """Class to model trained
    :external+sklearn:py:class:`sklearn.neural_network.MLPRegressor` or
    :external+sklearn:py:class:`sklearn.neural_network.MLPClassifier` with PySCIPOpt.

    |ClassShort|
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars=None,
        unique_naming_prefix="",
        classification=False,
        **kwargs,
    ):
        SKgetter.__init__(self, predictor, **kwargs)
        self.classification = classification
        if self.classification and predictor.n_outputs_ <= 2:
            self.output_size = 1
        else:
            self.output_size = predictor.n_outputs_
        BaseNNConstr.__init__(
            self,
            scip_model,
            predictor,
            input_vars,
            output_vars,
            unique_naming_prefix,
            **kwargs,
        )

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to SCIP."""
        neural_net = self.predictor
        if neural_net.activation not in self.act_dict:
            raise NoModel(
                neural_net,
                f"No implementation for activation function {neural_net.activation}",
            )
        activation = neural_net.activation

        input_vars = self._input

        for i in range(neural_net.n_layers_ - 1):
            layer_coefs = neural_net.coefs_[i]
            layer_intercept = neural_net.intercepts_[i]

            # For last layer change activation
            if i == neural_net.n_layers_ - 2:
                activation = neural_net.out_activation_
                # Cheat here: In the classification case we don't care about the activation function as the highest
                # value will always have the highest probability and therefore be the selected class
                if self.classification:
                    activation = "identity"
                if self.classification:
                    output = create_vars(
                        self.scip_model,
                        (self.input.shape[0], layer_coefs.shape[-1]),
                        vtype="C",
                        lb=None,
                        name_prefix=self.unique_naming_prefix + f"layer_{i}",
                    )
                    self._created_vars.append(output)
                else:
                    output = self.output
            else:
                output = create_vars(
                    self.scip_model,
                    (self.input.shape[0], layer_coefs.shape[-1]),
                    vtype="C",
                    lb=None,
                    name_prefix=self.unique_naming_prefix + f"layer_{i}",
                )
                self._created_vars.append(output)

            layer = self.add_dense_layer(
                input_vars,
                layer_coefs,
                layer_intercept,
                activation,
                output,
                unique_naming_prefix=self.unique_naming_prefix + f"layer_{i}_",
                **kwargs,
            )
            input_vars = layer.output

        if self.classification:
            if neural_net.out_activation_ == "logistic":
                one_dim_center = 0
            else:
                one_dim_center = 0.5
            new_vars, new_cons = argmax_bound_formulation(
                self.scip_model,
                input_vars,
                self.output,
                self.unique_naming_prefix,
                one_dim_center=one_dim_center,
            )
            for added_var in new_vars:
                self._created_vars.append(added_var)
            for added_cons in new_cons:
                self._created_cons.append(added_cons)
