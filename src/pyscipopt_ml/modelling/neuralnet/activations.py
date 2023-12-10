"""Internal module to make MIP modeling of activation functions."""

import numpy as np
from pyscipopt import exp, quicksum


def add_identity_activation_constraint_layer(layer):
    """
    MIP model for identity activation on a layer

    Parameters
    ----------
    layer : AbstractNNLayer
        Layer to which activation is applied.

    Returns
    -------

    affine_cons : np.ndarray
        A numpy array containing the linear transformation constraints

    """

    n_samples = layer.input.shape[0]
    n_nodes_left = layer.input.shape[-1]
    n_nodes_right = layer.output.shape[-1]
    affine_cons = np.zeros((n_samples, n_nodes_right), dtype=object)

    # Perform some basic activity based bound propagation
    propagation_success, lbs, ubs = propagate_identity_bounds(
        layer, n_samples, n_nodes_left, n_nodes_right, False
    )

    for i in range(n_samples):
        for j in range(n_nodes_right):
            rhs = (
                quicksum(layer.coefs[k][j] * layer.input[i][k] for k in range(n_nodes_left))
                + layer.intercept[j]
            )
            name = layer.unique_naming_prefix + f"affine_{i}_{j}"
            affine_cons[i][j] = layer.scip_model.addCons(layer.output[i][j] == rhs, name=name)
            # Propagate bounds
            if propagation_success:
                if abs(lbs[i][j]) < 10**5:
                    output_lb = layer.output[i][j].getLbOriginal()
                    layer.scip_model.chgVarLb(layer.output[i][j], max(lbs[i][j], output_lb))
                if abs(ubs[i][j]) < 10**5:
                    output_ub = layer.output[i][j].getUbOriginal()
                    layer.scip_model.chgVarUb(layer.output[i][j], min(ubs[i][j], output_ub))

    return affine_cons


def add_relu_activation_constraint_layer(layer, slack, activation_only=True):
    """
    MIP model for ReLU activation on a layer

    Parameters
    ----------
    layer : AbstractNNLayer
        Layer to which activation is applied.

    slack : np.ndarray
        Slack variables that will be used in the SOS formulation

    activation_only : bool, optional
        Whether this layer should only feature as an activation layer, i.e., skip the affine transformation

    Returns
    -------

    cons_with_slack : np.ndarray
        A numpy array containing added constraints

    sos_cons : np.ndarray
        A numpy array containing added constraints

    """

    # Initialise values for easy access and create empty constraint arrays
    n_samples = layer.input.shape[0]
    n_nodes_left = layer.input.shape[-1]
    n_nodes_right = layer.output.shape[-1]
    sos_cons = np.zeros((n_samples, n_nodes_right), dtype=object)
    cons_with_slack = np.zeros((n_samples, n_nodes_right), dtype=object)

    # Perform some basic activity based bound propagation
    propagation_success, lbs, ubs = propagate_identity_bounds(
        layer, n_samples, n_nodes_left, n_nodes_right, activation_only
    )

    # Iterate over all nodes on the right hand side and create the appropriate constraints
    for i in range(n_samples):
        for j in range(n_nodes_right):
            if layer.output[i][j].getLbOriginal() < 0:
                layer.scip_model.chgVarLb(layer.output[i][j], 0)
            name = layer.unique_naming_prefix + f"slack_{i}_{j}"
            if activation_only:
                cons_with_slack[i][j] = layer.scip_model.addCons(
                    layer.output[i][j] == layer.input[i][j] + slack[i][j], name=name
                )
            else:
                rhs = quicksum(layer.coefs[k][j] * layer.input[i][k] for k in range(n_nodes_left))
                rhs += layer.intercept[j] + slack[i][j]
                cons_with_slack[i][j] = layer.scip_model.addCons(
                    layer.output[i][j] == rhs, name=name
                )
            # Propagate bounds
            if propagation_success:
                if abs(lbs[i][j]) < 10**5:
                    output_lb = layer.output[i][j].getLbOriginal()
                    layer.scip_model.chgVarLb(
                        layer.output[i][j], max(max(lbs[i][j], 0), output_lb)
                    )
                    layer.scip_model.chgVarUb(slack[i][j], max(-lbs[i][j], 0))
                if abs(ubs[i][j]) < 10**5:
                    output_ub = layer.output[i][j].getUbOriginal()
                    layer.scip_model.chgVarUb(
                        layer.output[i][j], min(max(ubs[i][j], 0), output_ub)
                    )
                    layer.scip_model.chgVarLb(slack[i][j], max(-ubs[i][j], 0))
            name = layer.unique_naming_prefix + f"sos_{i}_{j}"
            sos_cons[i][j] = layer.scip_model.addConsSOS1(
                [layer.output[i][j], slack[i][j]], name=name
            )

    return cons_with_slack, sos_cons


def add_sigmoid_activation_constraint_layer(layer, activation_only=True):
    """
    MIP model for Sigmoid activation on a layer

    Parameters
    ----------
    layer : AbstractNNLayer
        Layer to which activation is applied.

    activation_only : bool, optional
        Whether this layer should only feature as an activation layer, i.e., skip the affine transformation

    Returns
    -------

    sigmoid_cons : np.ndarray
        A numpy array containing added constraints

    """

    # Initialise values for easy access and create empty constraint arrays
    n_samples = layer.input.shape[0]
    n_nodes_left = layer.input.shape[-1]
    n_nodes_right = layer.output.shape[-1]
    sigmoid_cons = np.zeros((n_samples, n_nodes_right), dtype=object)

    # Iterate over all nodes on the right hand side and create the appropriate constraints
    for i in range(n_samples):
        for j in range(n_nodes_right):
            if layer.output[i][j].getLbOriginal() < 0:
                layer.scip_model.chgVarLb(layer.output[i][j], 0)
            if layer.output[i][j].getUbOriginal() > 1:
                layer.scip_model.chgVarUb(layer.output[i][j], 1)
            if activation_only:
                x = layer.input[i][j]
            else:
                x = (
                    quicksum(layer.coefs[k][j] * layer.input[i][k] for k in range(n_nodes_left))
                    + layer.intercept[j]
                )
            name = layer.unique_naming_prefix + f"sigmoid_{i}_{j}"
            sigmoid_cons[i][j] = layer.scip_model.addCons(
                layer.output[i][j] == 1 / (1 + exp(-x)), name=name
            )

    return sigmoid_cons


def add_tanh_activation_constraint_layer(layer, activation_only=True):
    """
    MIP model for tanh activation on a layer

    Parameters
    ----------
    layer : AbstractNNLayer
        Layer to which activation is applied.

    activation_only : bool, optional
        Whether this layer should only feature as an activation layer, i.e., skip the affine transformation

    Returns
    -------

    tanh_cons : np.ndarray
        A numpy array containing added constraints

    """

    # Initialise values for easy access and create empty constraint arrays
    n_samples = layer.input.shape[0]
    n_nodes_left = layer.input.shape[-1]
    n_nodes_right = layer.output.shape[-1]
    tanh_cons = np.zeros((n_samples, n_nodes_right), dtype=object)

    # Iterate over all nodes on the right hand side and create the appropriate constraints
    for i in range(n_samples):
        for j in range(n_nodes_right):
            if layer.output[i][j].getLbOriginal() < -1:
                layer.scip_model.chgVarLb(layer.output[i][j], -1)
            if layer.output[i][j].getUbOriginal() > 1:
                layer.scip_model.chgVarUb(layer.output[i][j], 1)
            if activation_only:
                x = layer.input[i][j]
            else:
                x = (
                    quicksum(layer.coefs[k][j] * layer.input[i][k] for k in range(n_nodes_left))
                    + layer.intercept[j]
                )
            name = layer.unique_naming_prefix + f"tanh_{i}_{j}"
            tanh_cons[i][j] = layer.scip_model.addCons(
                layer.output[i][j] == (1 - exp(-2 * x)) / (1 + exp(-2 * x)), name=name
            )

    return tanh_cons


def propagate_identity_bounds(layer, n_samples, n_nodes_left, n_nodes_right, activation_only):
    """
    Activity based bound propagation. Assume the worst case bound for each node individually and generate
    bounds for the next layer.

    Parameters
    ----------
    layer : AbstractNNLayer
        Layer to which activation is applied.
    n_samples : int
        The number of samples passed by the user
    n_nodes_left : int
        The number of nodes on the left (input) layer
    n_nodes_right : int
        The number of nodes on the right (output) layer
    activation_only : bool
        Whether the bounds are going to be used for an activation only layer. In this case no weighted sum is needed.

    Returns
    -------
    propagation : bool
        Whether propagation worked
    lbs : np.ndarray
        An array of worst-case scenario lower bounds
    ubs : np.ndarray
        An array of worst-case scenario upper bounds
    """

    ubs = np.zeros((n_samples, n_nodes_right))
    lbs = np.zeros(
        (
            n_samples,
            n_nodes_right,
        )
    )
    input_lbs = np.zeros((n_samples, n_nodes_left))
    input_ubs = np.zeros((n_samples, n_nodes_left))
    for i in range(n_samples):
        for k in range(n_nodes_left):
            input_lbs[i][k] = layer.input[i][k].getLbOriginal()
            input_ubs[i][k] = layer.input[i][k].getUbOriginal()

    # In the case of activation only we can simply return the input bounds
    if activation_only:
        return True, input_lbs, input_ubs

    # Skip propagation for the weighted sum if some bounds are larger than 10**5
    if np.max(input_ubs) > 10**5 or np.min(input_lbs) < -1 * 10**5:
        return False, ubs, lbs

    for i in range(n_samples):
        for j in range(n_nodes_right):
            ub = 0
            lb = 0
            for k in range(n_nodes_left):
                coefficient = layer.coefs[k][j]
                if coefficient > 0:
                    ub += input_ubs[i][k] * coefficient
                    lb += input_lbs[i][k] * coefficient
                elif coefficient < 0:
                    ub += input_lbs[i][k] * coefficient
                    lb += input_ubs[i][k] * coefficient
            ubs[i][j] = ub + layer.intercept[j]
            lbs[i][j] = lb + layer.intercept[j]

    return True, lbs, ubs
