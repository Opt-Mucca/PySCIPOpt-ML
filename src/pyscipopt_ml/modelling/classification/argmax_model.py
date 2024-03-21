""" Utilities for modelling argmax constraints """

import numpy as np
from pyscipopt import quicksum

from ..var_utils import create_vars


def argmax_bound_formulation(scip_model, _input, output, unique_naming_prefix, one_dim_center=0.5):
    """
    Create constraints that represent the output of an argmax function applied to _input.
    The constraints ensure binary output of a single class.

    The formulation is different depending on the number of classes. In the case of there being two samples:

    Let x be the regression input \reals^{2}, and z the binary output {0, 1}^{2}
    .. math::

        \begin{align*}
        z_{1} : x_{1} >= x_{2}
        z_{2} : x_{2} >= x_{1}
        \sum z_{i} == 1
        \end{align*}

    for the case of arbitrary classes the formulation below is used:

    Let x be the regression input \reals^{n}, z the binary output
    {0, 1}^{n}, s the slack variables [0, inf]^{n}, and y the maximum over the input \reals:

    .. math::

        \begin{align*}
        x_{i} + s_{i} - y == 0 \forall i \in N
        SOS1(z_{i}, s_{i}) \forall i \in N
        \sum z_{i} == 1
        \end{align*}


    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    _input : np.ndarray
        The (potentially aggregated) output variables from the regression variant of a predictor, which are now
        input to the argmax formulation.
    output : np.ndarray
        The output variables of the (classification) predictor
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    one_dim_center : float, optional
        The value for which the 1-D argmax is centred around. Normally this is 0.5 for a single binary.
    Returns
    -------
    created_vars : list
        A list containing np.ndarray of PySCIPOpt variables that were created for the argmax formulation
    created_cons : list
        A list containing np.ndarray of PySCIPOpt constraints that we created for the argmax formulation
    """

    assert (
        _input.shape == output.shape
    ), f"Input and output dimensions do not match. {_input.shape} != {output.shape}"

    # get the in and out dimensions
    n_samples = _input.shape[0]
    outdim = output.shape[-1]

    # Separate the formulation into cases
    if outdim == 1:
        name_prefix = unique_naming_prefix + "argmax"
        bin_vars = create_vars(scip_model, shape=(n_samples,), vtype="B", name_prefix=name_prefix)

        # Create additional constraints
        output_equal_cons = np.zeros((n_samples,), dtype=object)
        output_under_half = np.zeros((n_samples,), dtype=object)
        output_over_half = np.zeros((n_samples,), dtype=object)

        # Now populate the constraints
        for i in range(n_samples):
            name = unique_naming_prefix + f"out_eq_{i}"
            output_equal_cons[i] = scip_model.addCons(output[i][0] == bin_vars[i], name=name)
            name = unique_naming_prefix + f"out_ub_{i}"
            output_under_half[i] = scip_model.addConsIndicator(
                _input[i][0] <= one_dim_center, bin_vars[i], activeone=False, name=name
            )
            name = unique_naming_prefix + f"out_lb_{i}"
            output_under_half[i] = scip_model.addConsIndicator(
                -_input[i][0] <= -one_dim_center, bin_vars[i], name=name
            )

        return [bin_vars], [output_equal_cons, output_under_half, output_over_half]

    elif outdim == 2:
        # Create additional variables
        name_prefix = unique_naming_prefix + "argmax"
        max_bin_vars = create_vars(
            scip_model, shape=(n_samples, outdim), vtype="B", name_prefix=name_prefix
        )

        # Create additional constraints
        output_equal_cons = np.zeros((n_samples, outdim), dtype=object)
        indicator_cons = np.zeros((n_samples, outdim), dtype=object)
        sum_bin_cons = np.zeros((n_samples,), dtype=object)

        # Now populate the constraints
        for i in range(n_samples):
            name = unique_naming_prefix + f"out_eq_{i}_0"
            output_equal_cons[i][0] = scip_model.addCons(
                output[i][0] == max_bin_vars[i][0], name=name
            )
            name = unique_naming_prefix + f"out_eq_{i}_1"
            output_equal_cons[i][1] = scip_model.addCons(
                output[i][1] == max_bin_vars[i][1], name=name
            )
            name = unique_naming_prefix + f"indicator_argmax_{i}_0"
            indicator_cons[i][0] = scip_model.addConsIndicator(
                -_input[i][0] <= -_input[i][1], max_bin_vars[i][0], name=name
            )
            name = unique_naming_prefix + f"indicator_argmax_{i}_1"
            indicator_cons[i][1] = scip_model.addConsIndicator(
                -_input[i][1] <= -_input[i][0], max_bin_vars[i][1], name=name
            )
            name = unique_naming_prefix + f"sum_bin_{i}"
            sum_bin_cons[i] = scip_model.addCons(
                quicksum(max_bin_vars[i][j] for j in range(outdim)) == 1, name=name
            )
        return [max_bin_vars], [output_equal_cons, indicator_cons, sum_bin_cons]
    else:
        # Create additional variables that are needed for classification
        name_prefix = unique_naming_prefix + "argmax"
        max_bin_vars = create_vars(
            scip_model, shape=(n_samples, outdim), vtype="B", name_prefix=name_prefix
        )
        name_prefix = unique_naming_prefix + "slack_argmax"
        slack_vars = create_vars(
            scip_model, shape=(n_samples, outdim), vtype="C", lb=0, name_prefix=name_prefix
        )
        name_prefix = unique_naming_prefix + "max_val"
        max_val_vars = create_vars(
            scip_model, shape=(n_samples,), vtype="C", lb=None, ub=None, name_prefix=name_prefix
        )

        # Create additional constraints that are needed for classification
        output_equal_cons = np.zeros((n_samples, outdim), dtype=object)
        sum_zero_cons = np.zeros((n_samples, outdim), dtype=object)
        sos_slack_bin_cons = np.zeros((n_samples, outdim), dtype=object)
        sum_bin_cons = np.zeros((n_samples,), dtype=object)

        for i in range(n_samples):
            for j in range(outdim):
                name = unique_naming_prefix + f"out_eq_{i}_{j}"
                output_equal_cons[i][j] = scip_model.addCons(
                    output[i][j] == max_bin_vars[i][j], name=name
                )
                name = unique_naming_prefix + f"slack_zero_eq_{i}_{j}"
                sum_zero_cons[i][j] = scip_model.addCons(
                    _input[i][j] + slack_vars[i][j] - max_val_vars[i] == 0, name=name
                )
                name = unique_naming_prefix + f"sos_slack_bin_{i}_{j}"
                sos_slack_bin_cons[i][j] = scip_model.addConsSOS1(
                    [slack_vars[i][j], max_bin_vars[i][j]], name=name
                )

            name = unique_naming_prefix + f"sum_bin_{i}"
            sum_bin_cons[i] = scip_model.addCons(
                quicksum(max_bin_vars[i][j] for j in range(outdim)) == 1, name=name
            )

        return [max_bin_vars, max_val_vars], [
            output_equal_cons,
            sum_zero_cons,
            sos_slack_bin_cons,
            sum_bin_cons,
        ]
