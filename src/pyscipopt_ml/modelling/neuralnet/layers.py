"""Bases classes for modeling neural network layers."""

from ...exceptions import ParameterError
from ..base_predictor_constraint import AbstractPredictorConstr
from ..var_utils import create_vars
from .activations import (
    add_identity_activation_constraint_layer,
    add_relu_activation_constraint_layer,
    add_sigmoid_activation_constraint_layer,
    add_tanh_activation_constraint_layer,
)


class AbstractNNLayer(AbstractPredictorConstr):
    """Abstract class for NN layers."""

    def __init__(
        self,
        scip_model,
        input_vars,
        output_vars,
        activation,
        unique_naming_prefix,
        **kwargs,
    ):
        self.activation = activation
        AbstractPredictorConstr.__init__(
            self, scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def get_error(self, eps=None):
        # We can't compute externally the error of a layer
        raise AssertionError("Cannot compute the error of an individual layer")


class ActivationLayer(AbstractNNLayer):
    """Class to build one activation layer of a neural network."""

    def __init__(
        self,
        scip_model,
        output_vars,
        input_vars,
        activation_function,
        unique_naming_prefix,
        **kwargs,
    ):
        super().__init__(
            scip_model,
            input_vars,
            output_vars,
            activation_function,
            unique_naming_prefix,
            **kwargs,
        )

    def _create_output_vars(self, input_vars):
        output_vars = create_vars(
            input_vars.shape,
            vtype="C",
            lb=None,
            ub=None,
            name_prefix=self.unique_naming_prefix + "output",
        )
        return output_vars

    def _mip_model(self, **kwargs):
        """Add the layer to model."""
        if self.activation == "relu":
            slack = create_vars(
                self.scip_model,
                (self.input.shape[0], self.output.shape[-1]),
                vtype="C",
                lb=0.0,
                ub=None,
                name_prefix=self.unique_naming_prefix + "slack",
            )
            affine_slack_cons, sos_cons = add_relu_activation_constraint_layer(
                self, slack, activation_only=True
            )
            self._created_vars.append(slack)
            self._created_cons.append(affine_slack_cons)
            self._created_cons.append(sos_cons)
        elif self.activation == "logistic" or self.activation == "sigmoid":
            sigmoid_cons = add_sigmoid_activation_constraint_layer(self, activation_only=True)
            self._created_cons.append(sigmoid_cons)
        elif self.activation == "tanh":
            tanh_cons = add_tanh_activation_constraint_layer(self, activation_only=True)
            self._created_cons.append(tanh_cons)
        else:
            raise ParameterError(f"Activation layer of type {self.activation} shouldn't exist")


class DenseLayer(AbstractNNLayer):
    """Class to build one layer of a neural network."""

    def __init__(
        self,
        scip_model,
        input_vars,
        layer_coefs,
        layer_intercept,
        output_vars,
        activation_function,
        unique_naming_prefix,
        **kwargs,
    ):
        self.coefs = layer_coefs
        self.intercept = layer_intercept
        super().__init__(
            scip_model,
            input_vars,
            output_vars,
            activation_function,
            unique_naming_prefix,
            **kwargs,
        )

    def _create_output_vars(self, input_vars):
        output_vars = create_vars(
            (input_vars.shape[0], self.coefs.shape[-1]),
            vtype="C",
            lb=None,
            ub=None,
            name_prefix=self.unique_naming_prefix + "output",
        )
        return output_vars

    def _mip_model(self, **kwargs):
        """Add the layer to model."""
        if self.activation == "relu":
            slack = create_vars(
                self.scip_model,
                (self.input.shape[0], self.output.shape[-1]),
                vtype="C",
                lb=0.0,
                ub=None,
                name_prefix=self.unique_naming_prefix + "slack",
            )
            affine_slack_cons, sos_cons = add_relu_activation_constraint_layer(
                self, slack, activation_only=False
            )
            self._created_vars.append(slack)
            self._created_cons.append(affine_slack_cons)
            self._created_cons.append(sos_cons)
        elif self.activation == "logistic" or self.activation == "sigmoid":
            sigmoid_cons = add_sigmoid_activation_constraint_layer(self, activation_only=False)
            self._created_cons.append(sigmoid_cons)
        elif self.activation == "tanh":
            tanh_cons = add_tanh_activation_constraint_layer(self, activation_only=False)
            self._created_cons.append(tanh_cons)
        elif self.activation == "identity":
            affine_cons = add_identity_activation_constraint_layer(self)
            self._created_vars.append(affine_cons)
        else:
            raise ParameterError(f"Activation layer of type {self.activation} shouldn't exist")
