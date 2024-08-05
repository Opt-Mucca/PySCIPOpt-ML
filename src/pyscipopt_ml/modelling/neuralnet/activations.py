"""Internal module to make MIP modeling of activation functions."""

import numpy as np
from pyscipopt import exp, log, quicksum


def add_identity_activation_constraint_layer(layer, max_bound):
    """
    MIP model for identity activation on a layer

    Parameters
    ----------
    layer : AbstractNNLayer
        Layer to which activation is applied.

    max_bound : float or int
        The maximum bound for which propagation values will be stored

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
        layer, n_samples, n_nodes_left, n_nodes_right, False, max_bound
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
                if abs(lbs[i][j]) < max_bound:
                    output_lb = layer.output[i][j].getLbOriginal()
                    layer.scip_model.chgVarLb(layer.output[i][j], max(lbs[i][j], output_lb))
                if abs(ubs[i][j]) < max_bound:
                    output_ub = layer.output[i][j].getUbOriginal()
                    layer.scip_model.chgVarUb(layer.output[i][j], min(ubs[i][j], output_ub))

    return affine_cons


def add_relu_activation_constraint_layer(layer, aux_vars, activation_only=True, formulation="sos"):
    """
    MIP model for ReLU activation on a layer

    Parameters
    ----------
    layer : AbstractNNLayer
        Layer to which activation is applied.

    aux_vars : np.ndarray
        Auxiliary variables that are used for the formulation. These are slack variables for the SOS formulation
        and binary activation variables for the Big-M formulation

    activation_only : bool, optional
        Whether this layer should only feature as an activation layer, i.e., skip the affine transformation

    formulation : str, optional
        The MIP formulation used for encoding the ReLU activation. Options are ["sos", "bigm"]

    Returns
    -------

    relu_cons : list
        A list of numpy arrays containing added constraints

    """

    # Initialise values for easy access and create empty constraint arrays
    n_samples = layer.input.shape[0]
    n_nodes_left = layer.input.shape[-1]
    n_nodes_right = layer.output.shape[-1]
    sos_cons = np.zeros((n_samples, n_nodes_right), dtype=object)
    cons_with_slack = np.zeros((n_samples, n_nodes_right), dtype=object)
    big_m_lb = np.zeros((n_samples, n_nodes_right), dtype=object)
    big_m_ub_inactive = np.zeros((n_samples, n_nodes_right), dtype=object)
    big_m_ub_active = np.zeros((n_samples, n_nodes_right), dtype=object)

    # Perform some basic activity based bound propagation
    max_bound = 10**5 if formulation == "sos" else np.inf
    propagation_success, lbs, ubs = propagate_identity_bounds(
        layer, n_samples, n_nodes_left, n_nodes_right, activation_only, layer.scip_model.infinity()
    )

    # Iterate over all nodes on the right hand side and create the appropriate constraints
    for i in range(n_samples):
        for j in range(n_nodes_right):
            # Propagate bounds
            if propagation_success:
                if abs(lbs[i][j]) < max_bound:
                    output_lb = layer.output[i][j].getLbOriginal()
                    layer.scip_model.chgVarLb(
                        layer.output[i][j], max(max(lbs[i][j], 0), output_lb)
                    )
                    if formulation == "sos":
                        layer.scip_model.chgVarUb(aux_vars[i][j], max(-lbs[i][j], 0))
                if abs(ubs[i][j]) < max_bound:
                    output_ub = layer.output[i][j].getUbOriginal()
                    layer.scip_model.chgVarUb(
                        layer.output[i][j], min(max(ubs[i][j], 0), output_ub)
                    )
                    if formulation == "sos":
                        layer.scip_model.chgVarLb(aux_vars[i][j], max(-ubs[i][j], 0))
            # Create layer constraints
            if layer.output[i][j].getLbOriginal() < 0:
                layer.scip_model.chgVarLb(layer.output[i][j], 0)
            if formulation == "sos":
                name = layer.unique_naming_prefix + f"slack_{i}_{j}"
                if activation_only:
                    cons_with_slack[i][j] = layer.scip_model.addCons(
                        layer.output[i][j] == layer.input[i][j] + aux_vars[i][j], name=name
                    )
                else:
                    rhs = quicksum(
                        layer.coefs[k][j] * layer.input[i][k] for k in range(n_nodes_left)
                    )
                    rhs += layer.intercept[j] + aux_vars[i][j]
                    cons_with_slack[i][j] = layer.scip_model.addCons(
                        layer.output[i][j] == rhs, name=name
                    )
                name = layer.unique_naming_prefix + f"sos_{i}_{j}"
                sos_cons[i][j] = layer.scip_model.addConsSOS1(
                    [layer.output[i][j], aux_vars[i][j]], name=name
                )
            else:
                if activation_only:
                    rhs = layer.input[i][j]
                else:
                    rhs = (
                        quicksum(
                            layer.coefs[k][j] * layer.input[i][k] for k in range(n_nodes_left)
                        )
                        + layer.intercept[j]
                    )
                name = layer.unique_naming_prefix + f"relu_lb_{i}_{j}"
                big_m_lb[i][j] = layer.scip_model.addCons(layer.output[i][j] >= rhs, name=name)
                name = layer.unique_naming_prefix + f"relu_ub_inactive_{i}_{j}"
                big_m_ub_inactive[i][j] = layer.scip_model.addCons(
                    layer.output[i][j] <= rhs - (1 - aux_vars[i][j]) * lbs[i][j], name=name
                )
                name = layer.unique_naming_prefix + f"relu_ub_active_{i}_{j}"
                big_m_ub_active[i][j] = layer.scip_model.addCons(
                    layer.output[i][j] <= aux_vars[i][j] * ubs[i][j], name=name
                )

    if formulation == "bigm":
        return [big_m_lb, big_m_ub_inactive, big_m_ub_active]
    else:
        return [cons_with_slack, sos_cons]


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


def add_softmax_activation_constraint_layer(layer, activation_only=True):
    """
    MIP model for softmax activation on a layer

    Parameters
    ----------
    layer : AbstractNNLayer
        Layer to which activation is applied.

    activation_only : bool, optional
        Whether this layer should only feature as an activation layer, i.e., skip the affine transformation

    Returns
    -------

    softmax_cons : np.ndarray
        A numpy array containing added constraints

    """

    # Initialise values for easy access and create empty constraint arrays
    n_samples = layer.input.shape[0]
    n_nodes_left = layer.input.shape[-1]
    n_nodes_right = layer.output.shape[-1]
    softmax_cons = np.zeros((n_samples, n_nodes_right), dtype=object)

    # Iterate over all nodes on the right hand side and create the appropriate constraints
    for i in range(n_samples):
        sum_exp_expr = 0
        for j in range(n_nodes_right):
            if activation_only:
                sum_exp_expr += exp(layer.input[i][j])
            else:
                sum_exp_expr += exp(
                    quicksum(layer.coefs[k][j] * layer.input[i][k] for k in range(n_nodes_left))
                    + layer.intercept[j]
                )
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
            name = layer.unique_naming_prefix + f"softmax_{i}_{j}"
            softmax_cons[i][j] = layer.scip_model.addCons(
                layer.output[i][j] == exp(x) / sum_exp_expr, name=name
            )

    return softmax_cons


def add_softplus_activation_constraint_layer(layer, activation_only=True):
    """
    MIP model for softplus activation on a layer

    Parameters
    ----------
    layer : AbstractNNLayer
        Layer to which activation is applied.

    activation_only : bool, optional
        Whether this layer should only feature as an activation layer, i.e., skip the affine transformation

    Returns
    -------

    softplus_cons : np.ndarray
        A numpy array containing added constraints

    """

    # Initialise values for easy access and create empty constraint arrays
    n_samples = layer.input.shape[0]
    n_nodes_left = layer.input.shape[-1]
    n_nodes_right = layer.output.shape[-1]
    softplus_cons = np.zeros((n_samples, n_nodes_right), dtype=object)

    # Iterate over all nodes on the right hand side and create the appropriate constraints
    for i in range(n_samples):
        for j in range(n_nodes_right):
            if layer.output[i][j].getLbOriginal() < 0:
                layer.scip_model.chgVarLb(layer.output[i][j], 0)
            if activation_only:
                x = layer.input[i][j]
            else:
                x = (
                    quicksum(layer.coefs[k][j] * layer.input[i][k] for k in range(n_nodes_left))
                    + layer.intercept[j]
                )
            name = layer.unique_naming_prefix + f"softplus_{i}_{j}"
            softplus_cons[i][j] = layer.scip_model.addCons(
                layer.output[i][j] == log(1 + exp(x)), name=name
            )

    return softplus_cons


def propagate_identity_bounds(
    layer, n_samples, n_nodes_left, n_nodes_right, activation_only, max_bound
):
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
    max_bound : float or int
        The maximum absolute bound value for a variable before bound propagation termination

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

    # Skip propagation for the weighted sum if some bounds are larger than max_bound
    if np.max(np.abs(input_ubs)) + np.max(np.abs(input_lbs)) > max_bound:
        return False, ubs, lbs

    for i in range(n_samples):
        for j in range(n_nodes_right):
            ub = 0
            lb = 0
            for k in range(n_nodes_left):
                coefficient = layer.coefs[k][j]
                if coefficient > 0:
                    ub += input_ubs[i][k] * coefficient + 10**-6
                    lb += input_lbs[i][k] * coefficient - 10**-6
                elif coefficient < 0:
                    ub += input_lbs[i][k] * coefficient + 10**-6
                    lb += input_ubs[i][k] * coefficient - 10**-6
            ubs[i][j] = ub + layer.intercept[j] + 10**-6
            lbs[i][j] = lb + layer.intercept[j] - 10**-6

    return True, lbs, ubs
