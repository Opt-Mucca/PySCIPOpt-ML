""" Utilities for modelling max constraints """

import numpy as np
from pyscipopt import quicksum

from ..var_utils import create_vars


def max_formulation(scip_model, _input, output, unique_naming_prefix):
    """
    Create constraints that represent the output of a max function applied to _input.

    The formulation is different depending on the number of features. In the case of there being one:

    Let x be the input \reals, and y the output also in \reals

    .. math::

        \begin{align*}
        y = x
        \end{align*}

    For the case of an arbitrary amount of features, the formulation below is used:

    Let x be the input \reals^{n}, y the output of the max function \reals,
    z be binary variables {0,1}^n,
    and s be the slack variables [0, inf]^{n}

    .. math::

        \begin{align*}
        x_{i} + s_{i} - y == 0 \forall i \in N
        SOS1(s_{i}, z_{i}) \forall i \in N
        \sum z_{i} >= 1
        \end{align*}


    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    _input : np.ndarray
        The (potentially aggregated) output variables from the regression variant of a predictor, which are now
        input to the max formulation.
    output : np.ndarray
        The output variables of the max function
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    Returns
    -------
    created_vars : list
        A list containing np.ndarray of PySCIPOpt variables that were created for the max formulation
    created_cons : list
        A list containing np.ndarray of PySCIPOpt constraints that we created for the max formulation
    """

    assert (output.shape[-1] == 1) and (
        _input.shape[0] == output.shape[0]
    ), f"Input and output dimensions are incorrect. Input: {_input.shape}. Output: {output.shape}"

    # get the in dimensions
    n_samples = _input.shape[0]
    n_features = _input.shape[-1]

    # Separate the formulation into cases
    if n_features == 1:
        name_prefix = unique_naming_prefix + "max"

        # Create additional constraints
        output_equal_cons = np.zeros((n_samples,), dtype=object)

        # Now populate the constraints
        for i in range(n_samples):
            name = unique_naming_prefix + f"out_eq_{i}"
            output_equal_cons[i] = scip_model.addCons(output[i][0] == _input[i][0], name=name)

        return [], [output_equal_cons]
    else:
        # Create additional variables that are needed for the max formulation
        name_prefix = unique_naming_prefix + "bin_max"
        max_bin_vars = create_vars(
            scip_model, shape=(n_samples, n_features), vtype="B", name_prefix=name_prefix
        )
        name_prefix = unique_naming_prefix + "slack_max"
        slack_vars = create_vars(
            scip_model, shape=(n_samples, n_features), vtype="C", lb=0, name_prefix=name_prefix
        )

        # Create additional constraints that are needed for the max formulation
        max_slack_equal_cons = np.zeros((n_samples, n_features), dtype=object)
        sos_slack_max_cons = np.zeros((n_samples, n_features), dtype=object)
        sum_bin_cons = np.zeros((n_samples,), dtype=object)

        for i in range(n_samples):
            for j in range(n_features):
                name = unique_naming_prefix + f"max_slack_eq_{i}_{j}"
                max_slack_equal_cons[i][j] = scip_model.addCons(
                    output[i][0] == _input[i][j] + slack_vars[i][j], name=name
                )
                name = unique_naming_prefix + f"sos_slack_max_{i}_{j}"
                sos_slack_max_cons[i][j] = scip_model.addConsSOS1(
                    [slack_vars[i][j], max_bin_vars[i][j]], name=name
                )

            name = unique_naming_prefix + f"sum_bin_max_{i}"
            sum_bin_cons[i] = scip_model.addCons(
                quicksum(max_bin_vars[i][j] for j in range(n_features)) >= 1, name=name
            )

        return [max_bin_vars, slack_vars], [max_slack_equal_cons, sos_slack_max_cons, sum_bin_cons]
