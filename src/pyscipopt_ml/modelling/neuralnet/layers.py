"""Bases classes for modeling neural network layers."""
import numpy as np

from ...exceptions import ParameterError
from ..base_predictor_constraint import AbstractPredictorConstr
from ..var_utils import create_vars
from .activations import (
    add_identity_activation_constraint_layer,
    add_relu_activation_constraint_layer,
    add_sigmoid_activation_constraint_layer,
    add_softmax_activation_constraint_layer,
    add_softplus_activation_constraint_layer,
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
        if "formulation" in kwargs:
            if kwargs["formulation"] not in ["sos", "bigm"]:
                formulation = kwargs["formulation"]
                raise ParameterError(f"Formulation type {formulation} is neither sos nor bigm")
            self.formulation = kwargs["formulation"]
        else:
            self.formulation = "sos"
        AbstractPredictorConstr.__init__(
            self, scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def get_error(self, eps=None):
        # We can't compute externally the error of a layer
        raise AssertionError("Cannot compute the error of an individual layer")

    def _layer_mip_model(self, activation_only=True, **kwargs):
        """Add the layer to model."""
        if self.activation == "relu":
            if self.formulation == "sos":
                slack = create_vars(
                    self.scip_model,
                    (self.input.shape[0], self.output.shape[-1]),
                    vtype="C",
                    lb=0.0,
                    ub=None,
                    name_prefix=self.unique_naming_prefix + "slack",
                )
                relu_cons = add_relu_activation_constraint_layer(
                    self, slack, activation_only=activation_only
                )
                self._created_vars.append(slack)
            else:
                activation_vars = create_vars(
                    self.scip_model,
                    (self.input.shape[0], self.output.shape[-1]),
                    vtype="B",
                    lb=0,
                    ub=1,
                    name_prefix=self.unique_naming_prefix + "relu_act",
                )
                relu_cons = add_relu_activation_constraint_layer(
                    self, activation_vars, activation_only=activation_only, formulation="bigm"
                )
                self._created_vars.append(activation_vars)
            for cons in relu_cons:
                self._created_cons.append(cons)
        elif self.activation == "logistic" or self.activation == "sigmoid":
            sigmoid_cons = add_sigmoid_activation_constraint_layer(
                self, activation_only=activation_only
            )
            self._created_cons.append(sigmoid_cons)
        elif self.activation == "tanh":
            tanh_cons = add_tanh_activation_constraint_layer(self, activation_only=activation_only)
            self._created_cons.append(tanh_cons)
        elif self.activation == "softmax":
            softmax_cons = add_softmax_activation_constraint_layer(
                self, activation_only=activation_only
            )
            self._created_cons.append(softmax_cons)
        elif self.activation == "softplus":
            softplus_cons = add_softplus_activation_constraint_layer(
                self, activation_only=activation_only
            )
            self._created_cons.append(softplus_cons)
        elif self.activation == "identity" and not activation_only:
            max_bound = self.scip_model.infinity() if self.formulation == "bigm" else 10**5
            affine_cons = add_identity_activation_constraint_layer(self, max_bound)
            self._created_vars.append(affine_cons)
        else:
            if activation_only:
                raise ParameterError(f"Activation layer of type {self.activation} shouldn't exist")
            else:
                raise ParameterError(f"Dense layer of type {self.activation} shouldn't exist")


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
        self._layer_mip_model(activation_only=True, **kwargs)


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
        self.intercept = np.array(self.intercept).reshape(-1)
        self._layer_mip_model(activation_only=False, **kwargs)
