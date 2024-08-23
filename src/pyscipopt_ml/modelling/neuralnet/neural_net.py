"""Bases classes for modeling neural network layers."""

from ..base_predictor_constraint import AbstractPredictorConstr
from .activations import (
    add_identity_activation_constraint_layer,
    add_relu_activation_constraint_layer,
    add_sigmoid_activation_constraint_layer,
    add_softmax_activation_constraint_layer,
    add_softplus_activation_constraint_layer,
    add_tanh_activation_constraint_layer,
)
from .layers import ActivationLayer, DenseLayer


class BaseNNConstr(AbstractPredictorConstr):
    """Base class for inserting a neural network into SCIP."""

    def __init__(
        self, scip_model, predictor, input_vars, output_vars, unique_naming_prefix, **kwargs
    ):
        self.predictor = predictor
        self.act_dict = {
            "relu": add_relu_activation_constraint_layer,
            "identity": add_identity_activation_constraint_layer,
            "tanh": add_tanh_activation_constraint_layer,
            "logistic": add_sigmoid_activation_constraint_layer,
            "softmax": add_softmax_activation_constraint_layer,
            "softplus": add_softplus_activation_constraint_layer,
        }
        self._layers = []

        super().__init__(scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs)

    def __iter__(self):
        return self._layers.__iter__()

    def add_dense_layer(
        self,
        input_vars,
        layer_coefs,
        layer_intercept,
        activation,
        output_vars=None,
        unique_naming_prefix="",
        **kwargs,
    ):
        """Add a dense layer to PySCIPOpt Model.

        Parameters
        ----------

        input_vars : np.ndarray
            Decision variables used as input for predictor in model.
        layer_coefs:
            Coefficient for each node in a layer
        layer_intercept:
            Intercept bias
        activation: str
            Type of activation used in layer
        output_vars : np.ndarray or None, optional
            Output variables
        unique_naming_prefix : str, optional
            A unique naming prefix that is used before all variable and constraint names.
        """
        layer = DenseLayer(
            self.scip_model,
            input_vars,
            layer_coefs,
            layer_intercept,
            output_vars,
            activation,
            unique_naming_prefix,
            **kwargs,
        )
        self._layers.append(layer)
        return layer

    def add_activation_layer(
        self, input_vars, activation, activation_vars=None, unique_naming_prefix="", **kwargs
    ):
        """Add an activation layer to the SCIP model.

        Parameters
        ----------

        input_vars : np.ndarray
            Decision variables used as input for predictor in SCIP model.
        activation: str
            Type of activation function used in layer
        activation_vars : np.ndarray, optional
            Output variables
        unique_naming_prefix : str, optional
            A unique naming prefix that is used before all variable and constraint names.
        """
        layer = ActivationLayer(
            self.scip_model,
            activation_vars,
            input_vars,
            activation,
            unique_naming_prefix,
            **kwargs,
        )
        self._layers.append(layer)
        return layer
