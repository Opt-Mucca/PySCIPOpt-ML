""" Utility functions for creating and verifying variables"""

import numpy as np

from ..exceptions import NoSolution


def create_vars(scip_model, shape, vtype, lb=None, ub=None, name_prefix=""):
    """
    Create PySCIPOpt variables in a numpy.ndarray of a given shape.

    Parameters
    ----------
     scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    shape : tuple
        The shape of the numpy array that will be constructed
    vtype : 'C' | 'B' | 'I'
        Whether the variables will be continuous, binary, or integer
    lb : float or int or None, optional
        The lower bound of the variables
    ub : float or int or None, optional
        The upper bound of the variables
    name_prefix : str, optional
        The naming prefix used for these variables

    Returns
    -------
    scip_vars : np.ndarray
        A np.ndarray with shape (shape) that contains uniquely names variables all of which are the specified type
    """

    scip_vars = np.zeros(shape, dtype=object)
    it = np.nditer(scip_vars, flags=["multi_index", "refs_ok"])
    for _ in it:
        idx_list = str(it.multi_index).strip(")").strip("(").split(",")
        idx_string = ""
        for idx in idx_list:
            if idx == "":
                continue
            idx_string += f"_{int(idx)}"
        name = name_prefix + idx_string
        scip_vars[it.multi_index] = scip_model.addVar(vtype=vtype, lb=lb, ub=ub, name=name)
    return scip_vars


def get_var_values(scip_model, scip_vars):
    """
    Get PySCIPOpt variable values in solved model for given array of variables

    Parameters
    ----------
     scip_model : PySCIPOpt Model
        The SCIP Model where the predictor should be inserted.
    scip_vars : np.ndarray
        A numpy array containing PySCIPOpt variables

    Returns
    -------
    scip_var_values : np.ndarray
        A np.ndarray that contains the variable values in the optimised scip_model for scip_vars

    Raises
    ------
    NoSolution
        If SCIP has no solution (either was not optimised or is infeasible).
    """

    if scip_model.getNSols() == 0:
        raise NoSolution

    scip_var_vals = np.zeros(scip_vars.shape)
    it = np.nditer(scip_vars, flags=["multi_index", "refs_ok"])
    for _ in it:
        scip_var_vals[it.multi_index] = scip_model.getVal(scip_vars[it.multi_index])

    return scip_var_vals
