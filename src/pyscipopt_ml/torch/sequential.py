"""Module for formulating :external+torch:py:class:`torch.nn.Sequential` model in a PySCIPOPT Model.
"""

import numpy as np
import torch
from torch import nn

from ..exceptions import NoModel, NoSolution, ParameterError
from ..modelling.classification import argmax_bound_formulation
from ..modelling.neuralnet import BaseNNConstr
from ..modelling.var_utils import create_vars


def add_sequential_constr(
    scip_model,
    sequential_model,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    output_type="regression",
    **kwargs,
):
    """Formulate sequential_model into scip_model.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the sequential model should be inserted.

    sequential_model : :external+torch:py:class:`torch.nn.Sequential`
        The sequential model to insert as predictor.

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
        each node of the final layer is the value from the sequential torch model.

    Returns
    -------
    SequentialConstr
        Object containing information about what was added to model to insert the
        predictor into it

    Raises
    ------
    NoModel
        If a section of the Pytorch model structure (layer or activation) is not implemented.

    Warning
    -------
    Only :external+torch:py:class:`torch.nn.Linear`,
    :external+torch:py:class:`torch.nn.ReLU`,
    :external+torch:py:class:`torch.nn.Sigmoid`,
    :external+torch:py:class:`torch.nn.Tanh`,
    :external+torch:py:class:`torch.nn.Softmax`, and
    :external+torch:py:class:`torch.nn.Softplus` layers are supported.

    Note
    ----
    |VariablesDimensionsWarn|

    """
    return SequentialConstr(
        scip_model,
        sequential_model,
        input_vars,
        output_vars,
        unique_naming_prefix,
        output_type,
        **kwargs,
    )


class SequentialConstr(BaseNNConstr):
    """Transform a pytorch Sequential Neural Network to SCIP constraints with
    input and output as matrices of variables.

    |ClassShort|.
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars=None,
        unique_naming_prefix="",
        output_type="regression",
        **kwargs,
    ):
        if output_type not in ["regression", "classification"]:
            raise NoModel(
                predictor, f"Output type {output_type} is neither regression nor classification"
            )
        self.output_type = output_type
        out_features = None
        n_steps = len(predictor)
        for i, step in enumerate(predictor):
            if (
                output_type == "classification"
                and i == n_steps - 1
                and not isinstance(step, nn.Linear)
            ):
                continue
            if isinstance(step, nn.ReLU):
                pass
            elif isinstance(step, nn.Sigmoid):
                pass
            elif isinstance(step, nn.Tanh):
                pass
            elif isinstance(step, nn.Flatten):
                pass
            elif isinstance(step, nn.Softmax):
                pass
            elif isinstance(step, nn.Softplus):
                pass
            elif isinstance(step, nn.Linear):
                out_features = step.out_features
                pass
            else:
                raise NoModel(predictor, f"Unsupported layer {type(step).__name__}")
        if out_features is not None:
            self.output_size = out_features
        else:
            raise NoModel(predictor, "There is no Linear Layer in the given NN")
        super().__init__(
            scip_model, predictor, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Add the predictor constraints to SCIP."""

        # Get the input variables and the number of layers in the torch sequential object
        input_vars = self.input
        num_layers = len(self.predictor)
        output = self.output

        # Iterate through each step of the predictor and create the appropriate layers
        for i, step in enumerate(self.predictor):
            # In the final layer's case use the actual output instead of some created intermediary output
            if i == num_layers - 1:
                output = self.output

            # In the case of activation functions create the appropriate layer.
            # Ignore for classification on the final layer. All supported activation functions preserve the maximum
            if (
                isinstance(step, nn.ReLU)
                or isinstance(step, nn.Sigmoid)
                or isinstance(step, nn.Tanh)
                or isinstance(step, nn.Softmax)
                or isinstance(step, nn.Softplus)
            ) and (i < num_layers - 1 or self.output_type == "regression"):
                if isinstance(step, nn.ReLU):
                    activation = "relu"
                elif isinstance(step, nn.Sigmoid):
                    activation = "logistic"
                elif isinstance(step, nn.Softmax):
                    activation = "softmax"
                elif isinstance(step, nn.Softplus):
                    activation = "softplus"
                else:
                    activation = "tanh"
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
                unique_naming_prefix = self.unique_naming_prefix + f"{activation}_{i}_"
                layer = self.add_activation_layer(
                    input_vars,
                    activation,
                    output,
                    unique_naming_prefix=unique_naming_prefix,
                    **kwargs,
                )
                input_vars = layer.output

            # In the case of a linear layer
            elif isinstance(step, nn.Linear):
                layer_weight, layer_bias = None, None
                for name, param in step.named_parameters():
                    if name == "weight":
                        layer_weight = param.detach().numpy().T
                    elif name == "bias":
                        layer_bias = param.detach().numpy()
                if layer_weight is None or layer_bias is None:
                    raise ParameterError(
                        "The torch sequential model linear layer contained no weights or bias!"
                    )
                if i < num_layers - 1 or self.output_type == "classification":
                    output = create_vars(
                        self.scip_model,
                        (input_vars.shape[0], layer_weight.shape[-1]),
                        vtype="C",
                        lb=None,
                        ub=None,
                        name_prefix=self.unique_naming_prefix + f"layer_{i}_",
                    )
                    self._created_vars.append(output)
                layer = self.add_dense_layer(
                    input_vars,
                    layer_weight,
                    layer_bias,
                    "identity",
                    output,
                    unique_naming_prefix=self.unique_naming_prefix + f"linear_{i}_",
                    **kwargs,
                )
                input_vars = layer.output

            # In the case of a Flatten layer just do nothing.
            elif isinstance(step, nn.Flatten):
                pass

            # In the case of classification force the output to be binary with the argmax formulation
            if i == num_layers - 1 and self.output_type == "classification":
                if isinstance(step, nn.Sigmoid):
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
        Returns error in SCIP's solution with respect to the actual output of the sequential neural network

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
            t_in = torch.from_numpy(self.input_values).float()
            t_out = self.predictor.forward(t_in)
            if self.output_type == "classification":
                t_out = nn.functional.one_hot(
                    torch.argmax(t_out, dim=1), num_classes=self.output_size
                )
            r_val = np.abs(t_out.detach().numpy() - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{t_out} != {self.output_values}")
            return r_val
        raise NoSolution()
