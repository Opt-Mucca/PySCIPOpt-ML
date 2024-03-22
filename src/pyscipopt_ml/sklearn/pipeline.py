"""Module for formulating a :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
into a PySCIPOpt Model.
The pipeline's transformers (or preprocessing steps) can be any of the following:
- :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler`
- :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures`
- :external+sklearn:py:class:`sklearn.preprocessing.Normalizer`
- :external+sklearn:py:class:`sklearn.preprocessing.Binarizer`

The final step of the pipeline must be a valid predictor.

"""

from ..exceptions import NoModel
from ..modelling.base_predictor_constraint import AbstractPredictorConstr
from ..modelling.get_convertor import get_convertor
from ..registered_predictors import sklearn_convertors
from .preprocessing import sklearn_transformers
from .skgetter import SKgetter


def add_pipeline_constr(
    scip_model, pipeline, input_vars, output_vars=None, unique_naming_prefix="", **kwargs
):
    """Formulate pipeline into scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    pipeline.

    Parameters
    ----------
    scip_model : SCIP Model
        The PySCIPOpt Model where the predictor should be inserted.
    pipeline : :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
        The pipeline to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for support vector in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for support vector in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    Returns
    -------
    PipelineConstr
        Object containing information about what was added to scip_model to embed the
        predictor into it

    Raises
    ------
    NoModel
        If the translation to SCIP of one of the elements in the pipeline
        is not implemented or recognized.

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return PipelineConstr(
        scip_model, pipeline, input_vars, output_vars, unique_naming_prefix, **kwargs
    )


class PipelineConstr(SKgetter, AbstractPredictorConstr):
    """Class to formulate a trained :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
    into a PySCIPOpt model.

    |ClassShort|
    """

    def __init__(
        self, scip_model, pipeline, input_vars, output_vars, unique_naming_prefix, **kwargs
    ):
        self._steps = []
        SKgetter.__init__(self, pipeline, **kwargs)
        AbstractPredictorConstr.__init__(
            self,
            scip_model,
            input_vars,
            output_vars,
            unique_naming_prefix,
            skip_validate=True,
            **kwargs,
        )

    def _mip_model(self, **kwargs):
        pipeline = self.predictor
        steps = self._steps
        input_vars = self._input

        for i, transformer in enumerate(pipeline[:-1]):
            convertor = get_convertor(transformer, sklearn_transformers())
            if convertor is None:
                raise NoModel(
                    self.predictor,
                    f"The sklearn pipeline transformer {transformer} is not supported",
                )
            steps.append(
                convertor(
                    self.scip_model,
                    transformer,
                    input_vars,
                    output_vars=None,
                    unique_naming_prefix=self.unique_naming_prefix + f"_step_{i}",
                    **kwargs,
                )
            )
            input_vars = steps[-1].output

        predictor = pipeline[-1]
        convertor = get_convertor(predictor, sklearn_convertors())
        if convertor is None:
            raise NoModel(
                self.predictor,
                f" The sklearn predictor {predictor} is not supported.",
            )
        steps.append(
            convertor(
                self.scip_model,
                predictor,
                input_vars,
                self._output,
                unique_naming_prefix=self.unique_naming_prefix + f"_pred_{len(steps) + 1}",
                **kwargs,
            )
        )
        if self._output is None:
            self._output = steps[-1].output
        self.output_size = steps[-1].output_size

        # As the output of the pipeline could not be validated before the individual steps were
        # created, we have to validate the model after creation.
        self._validate(self.input, self.output)

    @property
    def _has_solution(self):
        return self[-1]._has_solution

    @property
    def output(self):
        """Returns output variables of pipeline, i.e. output of its last step."""
        return self[-1].output

    @property
    def output_values(self):
        """Returns output values of pipeline in solution, i.e. output of its last step."""
        return self[-1].output_values

    @property
    def input(self):
        """Returns input variables of pipeline, i.e. input of its first step."""
        return self[0].input

    @property
    def input_values(self):
        """Returns input values of pipeline in solution, i.e. input of its first step."""
        return self[0].input_values

    def __getitem__(self, key):
        """Get an item from the pipeline steps."""
        return self._steps[key]

    def __iter__(self):
        """Iterate through pipeline steps."""
        return self._steps.__iter__()

    def __len__(self):
        """Get number of pipeline steps."""
        return self._steps.__len__()
