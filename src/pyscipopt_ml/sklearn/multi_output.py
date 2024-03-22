"""Module for formulating a :external+sklearn:py:class:`sklearn.multioutput.MultiOutputRegressor`
or :external+sklearn:py:class:`sklearn.multioutput.MultiOutputClassifier`
into a PySCIPOpt Model.

"""

from ..exceptions import NoModel
from ..modelling.base_predictor_constraint import AbstractPredictorConstr
from ..modelling.get_convertor import get_convertor
from ..registered_predictors import sklearn_convertors
from .skgetter import SKgetter


def add_multi_output_regressor_constr(
    scip_model,
    multi_output_regressor,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate a multi_output_regressor into scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    multi_output_regressor.

    Parameters
    ----------
    scip_model : SCIP Model
        The PySCIPOpt Model where the predictor should be inserted.
    multi_output_regressor : :external+sklearn:py:class:`sklearn.multioutput.MultiOutputRegressor`
        The multi_output_regressor to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for multi_output_regressor in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for multi_output_regressor in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    MultiOutputConstr
        Object containing information about what was added to scip_model to embed the
        predictor into it

    Raises
    ------
    NoModel
        If the translation to SCIP of one of the elements in the multi_output_regressor
        is not implemented or recognized.

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return MultiOutputConstr(
        scip_model,
        multi_output_regressor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        classification=False,
        **kwargs,
    )


def add_multi_output_classifier_constr(
    scip_model,
    multi_output_classifier,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    **kwargs,
):
    """Formulate a multi_output_classifier into scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    multi_output_classifier.

    Parameters
    ----------
    scip_model : SCIP Model
        The PySCIPOpt Model where the predictor should be inserted.
    multi_output_classifier : :external+sklearn:py:class:`sklearn.multioutput.MultiOutputClassifier`
        The multi_output_classifier to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for multi_output_classifier in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for multi_output_classifier in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    MultiOutputConstr
        Object containing information about what was added to scip_model to embed the
        predictor into it

    Raises
    ------
    NoModel
        If the translation to SCIP of one of the elements in the multi_output_classifier
        is not implemented or recognized.

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return MultiOutputConstr(
        scip_model,
        multi_output_classifier,
        input_vars,
        output_vars,
        unique_naming_prefix,
        classification=True,
        **kwargs,
    )


class MultiOutputConstr(SKgetter, AbstractPredictorConstr):
    """Class to formulate a trained :external+sklearn:py:class:`sklearn.multioutput.MultiOutputRegressor`
    or :external+sklearn:py:class:`sklearn.multioutput.MultiOutputClassifier`
    into a PySCIPOpt model.

    |ClassShort|
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        classification,
        **kwargs,
    ):
        self._estimators = []
        self.classification = classification
        self.output_size = len(predictor.estimators_)
        SKgetter.__init__(self, predictor, **kwargs)
        AbstractPredictorConstr.__init__(
            self,
            scip_model,
            input_vars,
            output_vars,
            unique_naming_prefix,
            **kwargs,
        )

    def _mip_model(self, **kwargs):
        for i, estimator in enumerate(self.predictor.estimators_):
            convertor = get_convertor(estimator, sklearn_convertors())
            if convertor is None:
                raise NoModel(
                    self.predictor,
                    f"The sklearn multi-output contains estimator {estimator}, which is not supported",
                )
            self._estimators.append(
                convertor(
                    self.scip_model,
                    estimator,
                    self.input,
                    self.output[:, i].reshape(-1, 1),
                    unique_naming_prefix=self.unique_naming_prefix + f"_step_{i}",
                    **kwargs,
                )
            )
