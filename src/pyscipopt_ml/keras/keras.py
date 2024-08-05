"""Module for formulating a `keras.Model <https://keras.io/api/models/model/>` into a PySCIPOpt Model."""

import pdb

import numpy as np
from tensorflow import keras

from ..exceptions import NoModel, NoSolution
from ..modelling.classification import argmax_bound_formulation
from ..modelling.neuralnet import BaseNNConstr
from ..modelling.var_utils import create_vars


def add_keras_constr(
    scip_model,
    keras_model,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    output_type="regression",
    **kwargs,
):
    """Formulate keras_model into scip_model.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the sequential model should be inserted.

    keras_model : `keras.Model <https://keras.io/api/models/model/>`
        The keras model to insert as predictor.

    input_vars : np.ndarray
        Decision variables used as input for the sequential neural network in PySCIPOpt Model.

    output_vars : np.ndarray
        Decision variables used as output for the sequential neural network in the PySCIPOpt Model

    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    output_type : {"classification", "regression"}, default="regression"
        If the option chosen is "classification" the output is 1 for exactly one class and
        0 for all others. If the option chosen is "regression" then the output for
        each node of the final layer is the value from the keras model.

    Returns
    -------
    KerasNetworkConstr
        Object containing information about what was added to scip_model to formulate
        keras_model into it

    Raises
    ------
    NoModel
        If the translation for some of the Keras model structure
        (layer or activation) is not implemented.

    Warnings
    --------
    Only `Dense <https://keras.io/api/layers/core_layers/dense/>`_
    (with `relu / tanh / sigmoid / softplus / softmax` activation) are supported.

    Notes
    -----
    |VariablesDimensionsWarn|

    """
    return KerasNetworkConstr(
        scip_model,
        keras_model,
        input_vars,
        output_vars,
        unique_naming_prefix,
        output_type,
        **kwargs,
    )


class KerasNetworkConstr(BaseNNConstr):
    """Transform a keras dense Neural Network to SCIP constraints with
    input and output as matrices of variables.

    |ClassShort|.
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        output_type,
        **kwargs,
    ):
        if output_type not in ["regression", "classification"]:
            raise NoModel(
                predictor, f"Output type {output_type} is neither regression nor classification"
            )
        self.output_type = output_type
        assert predictor.built
        out_features = None
        n_steps = len(predictor.layers)
        for i, step in enumerate(predictor.layers):
            if (
                output_type == "classification"
                and i == n_steps - 1
                and not isinstance(step, keras.layers.Dense)
            ):
                continue
            if isinstance(step, keras.layers.Dense):
                config = step.get_config()
                activation = config["activation"]
                if activation not in ("relu", "linear", "sigmoid", "tanh", "softplus", "softmax"):
                    raise NoModel(predictor, f"Unsupported activation {activation}")
                out_features = step.units
            elif isinstance(step, keras.layers.ReLU):
                if step.negative_slope != 0.0:
                    raise NoModel(predictor, "Only handle ReLU layers with negative slope 0.0")
                if step.threshold != 0.0:
                    raise NoModel(predictor, "Only handle ReLU layers with threshold of 0.0")
                if step.max_value is not None and step.max_value < float("inf"):
                    raise NoModel(predictor, "Only handle ReLU layers without maxvalue")
            elif isinstance(step, keras.layers.InputLayer):
                pass
            elif isinstance(step, keras.layers.Activation):
                activation = step.get_config()["activation"]
                if activation in ["sigmoid", "relu", "tanh", "softplus", "softmax"]:
                    pass
                else:
                    raise NoModel(predictor, f"Unsupported activation {activation}")
            else:
                raise NoModel(predictor, f"Unsupported network layer {type(step).__name__}")
        if out_features is not None:
            self.output_size = out_features

        super().__init__(
            scip_model, predictor, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Add the predictor constraints to SCIP."""

        # Get the input variables and the number of layers in the keras model object
        input_vars = self.input
        output = self.output
        num_layers = len(self.predictor.layers)

        # Iterate through each step of the predictor and create the appropriate layers
        for i, step in enumerate(self.predictor.layers):
            # In the final layer's case use the actual output instead of some created intermediary output
            if i == num_layers - 1:
                output = self._output

            # In the case of activation functions create the appropriate layer.
            # Ignore for classification on the final layer. All supported activation functions preserve the maximum
            if isinstance(step, keras.layers.InputLayer):
                pass
            elif (
                isinstance(step, keras.layers.ReLU) or isinstance(step, keras.layers.Activation)
            ) and (i < num_layers - 1 or self.output_type == "regression"):
                if i < num_layers - 1:
                    output = create_vars(
                        self.scip_model,
                        input_vars.shape,
                        vtype="C",
                        lb=None,
                        ub=None,
                        name_prefix=self.unique_naming_prefix + f"layer_{i}",
                    )
                    self._created_vars.append(output)
                if isinstance(step, keras.layers.ReLU):
                    activation = "relu"
                else:
                    activation = step.get_config()["activation"]
                unique_naming_prefix = self.unique_naming_prefix + f"{activation}_{i}_"
                layer = self.add_activation_layer(
                    input_vars,
                    activation,
                    output,
                    unique_naming_prefix=unique_naming_prefix,
                    **kwargs,
                )
                input_vars = layer.output

            # In the case of a dense layer
            elif isinstance(step, keras.layers.Dense):
                activation = step.get_config()["activation"]
                if activation == "linear" or (
                    i == num_layers - 1 and self.output_type == "classification"
                ):
                    activation = "identity"
                weights, bias = step.get_weights()
                if i < num_layers - 1:
                    output = create_vars(
                        self.scip_model,
                        (input_vars.shape[0], weights.shape[-1]),
                        vtype="C",
                        lb=None,
                        ub=None,
                        name_prefix=self.unique_naming_prefix + f"layer_{i}",
                    )
                    self._created_vars.append(output)
                unique_naming_prefix = self.unique_naming_prefix + f"{activation}_{i}_"
                layer = self.add_dense_layer(
                    input_vars,
                    weights,
                    bias,
                    activation,
                    output,
                    unique_naming_prefix=unique_naming_prefix,
                    **kwargs,
                )
                input_vars = layer.output

            # In the case of classification force the output to be binary with the argmax formulation
            if i == num_layers - 1 and self.output_type == "classification":
                if (
                    isinstance(step, keras.layers.Activation)
                    and step.get_config()["activation"] == "sigmoid"
                ):
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
                for var_set in new_vars:
                    self._created_vars.append(var_set)
                for cons_set in new_cons:
                    self._created_cons.append(cons_set)

    def get_error(self, eps=None):
        """
        Returns error in SCIP's solution with respect to the actual output of the keras model

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
            Using torch / pyscipopt, the absolute difference between model.forward(input) and scip.getVal(output).

        Raises
        ------
        NoSolution
            If SCIP has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution:
            t_out = self.predictor.predict(self.input_values)
            if self.output_type == "classification":
                t_out = keras.utils.to_categorical(
                    np.argmax(t_out, axis=1), num_classes=self.output_size
                )
            r_val = np.abs(t_out - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{t_out} != {self.output_values}")
            return r_val
        raise NoSolution()
