""" Utilities for modelling decision trees """
import pdb

import numpy as np
from pyscipopt import quicksum

from ..var_utils import create_vars


def compute_leafs_bounds(tree, epsilon, infinity):
    """Compute the bounds that define each leaf of the tree"""
    node_count = tree["node_count"]
    n_features = tree["n_features"]
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]

    node_lb = -np.ones((n_features, node_count)) * infinity
    node_ub = np.ones((n_features, node_count)) * infinity

    stack = [
        0,
    ]

    while len(stack) > 0:
        node = stack.pop()
        left = children_left[node]
        if left < 0:
            continue
        right = children_right[node]
        assert left not in stack
        assert right not in stack
        node_ub[:, right] = node_ub[:, node]
        node_lb[:, right] = node_lb[:, node]
        node_ub[:, left] = node_ub[:, node]
        node_lb[:, left] = node_lb[:, node]

        node_ub[feature[node], left] = threshold[node] - (epsilon / 2)
        node_lb[feature[node], right] = threshold[node] + (epsilon / 2)
        stack.append(right)
        stack.append(left)

    return node_lb, node_ub


def leaf_formulation(
    scip_model, _input, output, tree, unique_naming_prefix, epsilon, classification=False
):
    """Formulate decision tree using 'leaf' formulation

    We have one variable per leaf of the tree and a series of indicator constraints to
    define when that leaf is reached.

    The first step of the procedure is to derive input bounds for each leaf of the decision tree. These bounds will
    dictate for which input values the leaf can be reached. For a single sample, let x \reals^{nI} be the input,
    z {0,1}^{nL} be binary variables representing if a leaf is reached or not, and y \reals^{nO} be the output.

    .. math::

        \begin{align*}
        z_{i} -> x_{j} \geq leaf_lb[i][j] \forall i \in nL, j \in nI
        z_{i} -> x_{j} \leq leaf_ub[i][j] \forall i \in nL, j \in nI
        if classification:
            y_{j} == \sum z_{i} \forall i \in nL \text{, where class j is output of z_i}
            \sum y_{k} == 1
        else:
            z_{i} -> y_{k} = leaf_value[i][j] \forall i \in nL, k \in nO
        \sum z_{i} == 1
        \end{align*}

    """

    # Create names for items we want to access frequently
    n_samples = _input.shape[0]
    n_features = tree["n_features"]
    outdim = output.shape[-1]

    # Collect leaf nodes
    leaf_ids = tree["children_left"] <= -1
    n_leafs = sum(leaf_ids)
    name_prefix = unique_naming_prefix + "leaf"
    leaf_vars = create_vars(
        scip_model, shape=(n_samples, n_leafs), vtype="B", lb=0, name_prefix=name_prefix
    )

    # Calculate bounds for each leaf node
    (node_lb, node_ub) = compute_leafs_bounds(tree, epsilon, scip_model.infinity())

    # Create empty constraint objects
    output_class_sum_leaf_cons = np.zeros((n_samples, outdim), dtype=object)
    indicator_output_cons = np.zeros((n_samples, n_leafs, outdim, 2), dtype=object)
    indicator_leaf_lb = np.zeros((n_samples, n_leafs, n_features), dtype=object)
    indicator_leaf_ub = np.zeros((n_samples, n_leafs, n_features), dtype=object)

    # Iterate over all leaf nodes (They are the non-zero entries in leaf_ids)
    for i in range(n_samples):
        leafs_per_class = [0 for _ in range(outdim)]
        for j, node in enumerate(leaf_ids.nonzero()[0]):
            fixed_var = False
            # Fix the leaf variable to 0 if the input bounds do not allow the leaf to be reached
            for feature in range(n_features):
                if (
                    _input[i][feature].getLbOriginal() > node_ub[feature][node]
                    or _input[i][feature].getUbOriginal() < node_lb[feature][node]
                ):
                    scip_model.fixVar(leaf_vars[i][j], 0)
                    fixed_var = True
                    break
            # If the leaf could be reached, then add two sets of indicator constraints.
            # The first will enforce that a leaf node is only selected if the input values result in such a leaf.
            # The second force the appropriate value output by the leaf to be selected
            if not fixed_var:
                for feature in range(n_features):
                    name_lb = unique_naming_prefix + f"indicator_lb_{i}_{j}_{feature}"
                    name_ub = unique_naming_prefix + f"indicator_ub_{i}_{j}_{feature}"
                    feat_lb = node_lb[feature, node]
                    feat_ub = node_ub[feature, node]
                    if (
                        feat_lb > -scip_model.infinity()
                        and _input[i][feature].getLbOriginal() < feat_lb
                    ):
                        indicator_leaf_lb[i][j][feature] = scip_model.addConsIndicator(
                            -_input[i][feature] <= -feat_lb, leaf_vars[i][j], name=name_lb
                        )
                    if (
                        feat_ub < scip_model.infinity()
                        and _input[i][feature].getUbOriginal() > feat_ub
                    ):
                        indicator_leaf_ub[i][j][feature] = scip_model.addConsIndicator(
                            _input[i][feature] <= feat_ub, leaf_vars[i][j], name=name_ub
                        )
                # Iterate over the final output shape (num_outputs)
                # In the case of classification (num_classes), simply force the most frequent class to be selected
                if classification:
                    value = int(np.argmax(tree["value"][node][0]))
                    if outdim == 1:
                        if value == 1:
                            leafs_per_class[0] += leaf_vars[i][j]
                    else:
                        leafs_per_class[value] += leaf_vars[i][j]
                else:
                    for k in range(outdim):
                        name_ub = unique_naming_prefix + f"indicator_output_{i}_{j}_{k}_0"
                        name_lb = unique_naming_prefix + f"indicator_output_{i}_{j}_{k}_1"
                        value = tree["value"][node][k][0]
                        indicator_output_cons[i][j][k][0] = scip_model.addConsIndicator(
                            output[i][k] <= value, leaf_vars[i][j], name=name_ub
                        )
                        indicator_output_cons[i][j][k][1] = scip_model.addConsIndicator(
                            -output[i][k] <= -value, leaf_vars[i][j], name=name_lb
                        )
        # Add constraints that ensure the correct class is selected depending on the leaf
        if classification:
            for j in range(outdim):
                name = f"class_leaf_{i}_{j}"
                output_class_sum_leaf_cons[i][j] = scip_model.addCons(
                    output[i][j] == leafs_per_class[j], name=name
                )

    # Now add the constraints that only one leaf can be selected.
    # In the case of classification there is an additional constraint that only one class can be selected
    leaf_sum_cons = np.zeros(n_samples, dtype=object)
    for i in range(n_samples):
        name = unique_naming_prefix + f"sum_leafs_{i}"
        leaf_sum_cons[i] = scip_model.addCons(
            quicksum(leaf_vars[i][j] for j in range(leaf_vars.shape[-1])) == 1, name=name
        )

    # Finally set potentially stronger global bounds on the output variables (in the case of regression)
    if not classification:
        max_vals = [np.max(tree["value"][:, j, :]) for j in range(outdim)]
        min_vals = [np.min(tree["value"][:, j, :]) for j in range(outdim)]
        for i in range(n_samples):
            for j in range(outdim):
                if output[i][j].getLbOriginal() < min_vals[j]:
                    scip_model.chgVarLb(output[i][j], min_vals[j])
                if output[i][j].getUbOriginal() > max_vals[j]:
                    scip_model.chgVarUb(output[i][j], max_vals[j])

    # Now return the added constraints and variables
    if classification:
        return [leaf_vars], [
            indicator_leaf_lb,
            indicator_leaf_ub,
            output_class_sum_leaf_cons,
            leaf_sum_cons,
        ]
    else:
        return [leaf_vars], [
            indicator_leaf_lb,
            indicator_leaf_ub,
            indicator_output_cons,
            leaf_sum_cons,
        ]
