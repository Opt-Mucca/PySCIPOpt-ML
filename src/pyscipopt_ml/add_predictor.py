from warnings import warn

from .exceptions import NotRegistered
from .modelling.get_convertor import get_convertor
from .registered_predictors import registered_predictors


def add_predictor_constr(
    scip_model, predictor, input_vars, output_vars=None, unique_naming_prefix="p_", **kwargs
):
    """Formulate predictor in PySCIPOpt model.

    The formulation predicts the values of output_vars using input_vars according to
    predictor.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The pyscipopt model where the predictor should be inserted.
    predictor:
        The predictor to insert.
    input_vars : list or np.ndarray
        Decision variables used as input for predictor in scip_model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for predictor in scip_model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    AbstractPredictorConstr
        Object containing information about what was added to scip_model to insert the
        predictor in it

    Note
    ----
    The parameters `input_vars` and `output_vars` can be either

     * Lists of variables (List of lists etc. for higher dimensional input)
     * np.ndarray of variables

     For internal use in the package they are cast into a np.ndarray of variables

    They should have dimensions that conform with the input/output of the predictor.
    We denote by `n_samples` the number of samples (or objects) that we want to predict with our predictor.
    We denote by `n_features` the dimension of the input of the predictor.
    We denote by `n_output` the dimension of the output.

    The `input_vars` are therefore of shape `(n_samples, n_features)` and the `output_vars` of
    shape `(n_samples, n_outputs)`. In the case of `output_vars` not being passed, appropriate variables will
    be automatically created.
    In the case of `n_samples == 1` the first dimension can simply be removed from the input.
    """
    convertors = registered_predictors()
    convertor = get_convertor(predictor, convertors)
    if convertor is None:
        raise NotRegistered(type(predictor).__name__)
    if len(unique_naming_prefix) > 0 and unique_naming_prefix[0].isdigit():
        warn(
            f"Unique naming prefix {unique_naming_prefix} begins with a digit and is unsafe for printing LP files"
        )
    return convertor(
        scip_model, predictor, input_vars, output_vars, unique_naming_prefix, **kwargs
    )
