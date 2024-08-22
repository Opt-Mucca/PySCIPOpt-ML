from abc import ABC, abstractmethod

import numpy as np
from pyscipopt.scip import Constraint, Variable

from ..exceptions import NoSolution, ParameterError
from .var_utils import create_vars


class AbstractPredictorConstr(ABC):
    """Base class to store all information of embedded ML model by :py:func`pyscipopt_ml.add_predictor_constr`.

    This class is the base class to store everything that is added to
    a SCIP model when a trained predictor is inserted into it. Depending on
    the type of the predictor, a class derived from it will be returned
    by :py:func:`pyscipopt_ml.add_predictor_constr`.

    Warning
    -------

    Users should usually never construct objects of this class or one of its derived
    classes. They are returned by the :py:func:`pyscipopt_ml.add_predictor_constr` and
    other functions.
    """

    def __init__(
        self,
        scip_model,
        input_vars,
        output_vars=None,
        unique_naming_prefix="",
        skip_validate=False,
        **kwargs,
    ):
        self.scip_model = scip_model
        self.unique_naming_prefix = unique_naming_prefix
        self._validate(input_vars, output_vars, skip_output=skip_validate)
        self._created_vars = []
        self._created_cons = []
        self._build_predictor_model(**kwargs)

    def _validate(self, input_vars, output_vars=None, skip_output=False):
        """Validate input and output variables (check shapes, reshape if needed)."""

        # Ensure the correct type of input and output is given
        if type(input_vars) not in [list, np.ndarray]:
            raise ParameterError(
                f"Input variables are not type list or np.ndarray. They are type {type(input_vars)}."
            )
        if output_vars is not None:
            if not isinstance(output_vars, list) and not isinstance(output_vars, np.ndarray):
                raise ParameterError(
                    f"Output variables are not type list or np.ndarray. They are type {type(output_vars)}."
                )

        # Transform the type list to type np.ndarray
        if isinstance(input_vars, list):
            input_vars = np.array(input_vars, dtype=object)
        if isinstance(output_vars, list):
            output_vars = np.array(output_vars, dtype=object)

        # Change the dimension of the input variables if needed. (Always want number of data points first)
        if input_vars.ndim == 1:
            input_vars = input_vars.reshape((1, -1))
        if input_vars.ndim >= 3:
            input_vars = input_vars.reshape((input_vars.shape[0], -1))

        # Ensure that the input variable dimensions match that of the predictor
        if (
            hasattr(self, "input_size")
            and self.input_size is not None
            and input_vars.shape[-1] != self.input_size
        ):
            raise ParameterError(
                f"Input variables dimension don't conform with predictor {type(self)} "
                + f"Input variable dimensions: {input_vars.shape[-1]} != {self.input_size}"
            )

        # Skip output validation if requested
        if not skip_output:
            # In the case of the output being None, create the appropriate output variables here
            if output_vars is None:
                output_vars = self._create_output_vars(input_vars)

            # Change the dimensions of the output variables if needed (Always want the number of data points first)
            if output_vars.ndim == 1:
                if input_vars.shape[0] == 1:
                    output_vars = output_vars.reshape((1, -1))
                else:
                    output_vars = output_vars.reshape((-1, 1))

            # Ensure that the output variable dimensions match that of the predictor
            if (
                hasattr(self, "output_size")
                and self.output_size is not None
                and output_vars.shape[-1] != self.output_size
            ):
                raise ParameterError(
                    f"Output variable dimensions don't conform with predictor {type(self)} "
                    + f"Output variable dimensions: {output_vars.shape[-1]} != {self.output_size}"
                )

            if output_vars.shape[0] != input_vars.shape[0]:
                raise ParameterError(
                    "Non-conforming dimension between input variables and output variables: "
                    + f"{output_vars.shape[0]} != {input_vars.shape[0]}"
                )

        self._input = input_vars
        self._output = output_vars

    def _build_predictor_model(self, **kwargs):
        self._mip_model(**kwargs)

    def _get_created_vars_and_cons(self):
        created_vars = [c_vars for c_vars in self._created_vars]
        created_cons = [c_cons for c_cons in self._created_cons]
        if hasattr(self, "_estimators"):
            for estimator in self._estimators:
                sub_created_vars, sub_created_cons = estimator._get_created_vars_and_cons()
                created_vars += sub_created_vars
                created_cons += sub_created_cons
        if hasattr(self, "_layers"):
            for layer in self._layers:
                sub_created_vars, sub_created_cons = layer._get_created_vars_and_cons()
                created_vars += sub_created_vars
                created_cons += sub_created_cons
        if hasattr(self, "_steps"):
            for step in self._steps:
                sub_created_vars, sub_created_cons = step._get_created_vars_and_cons()
                created_vars += sub_created_vars
                created_cons += sub_created_cons

        return created_vars, created_cons

    def print_stats(self, file=None):
        """Print statistics on model additions stored by this class.

        This function prints detailed statistics on the variables
        and constraints that were added to the model.

        Arguments
        ---------

        file: None, optional
            Text stream to which output should be redirected. By default, this is sys.stdout.
        """

        n_indicator_cons = 0
        n_sos_cons = 0
        n_linear_cons = 0
        n_nonlinear_cons = 0

        created_vars, created_cons = self._get_created_vars_and_cons()
        for cons_set in created_cons:
            it = np.nditer(cons_set, flags=["multi_index", "refs_ok"])
            for _ in it:
                if isinstance(cons_set[it.multi_index], Constraint):
                    cons_type = cons_set[it.multi_index].getConshdlrName()
                    if cons_type == "indicator":
                        n_indicator_cons += 1
                    elif cons_type == "SOS1":
                        n_sos_cons += 1
                    elif cons_type == "linear":
                        n_linear_cons += 1
                    elif cons_type == "nonlinear":
                        n_nonlinear_cons += 1
                    else:
                        raise TypeError(
                            f"Cons {cons_set[it.multi_index]} is of unknown type {cons_type}"
                        )

        n_bin_vars = 0
        n_cont_vars = 0

        for var_set in created_vars:
            it = np.nditer(var_set, flags=["multi_index", "refs_ok"])
            for _ in it:
                if isinstance(var_set[it.multi_index], Variable):
                    var_type = var_set[it.multi_index].vtype()
                    if var_type == "BINARY":
                        n_bin_vars += 1
                    elif var_type == "CONTINUOUS":
                        n_cont_vars += 1
                    else:
                        raise TypeError(
                            f"Var {var_set[it.multi_index]} is of unknown type {var_type}"
                        )

        print(
            f"Constraints created:\n Linear    {n_linear_cons}\n Indicator {n_indicator_cons}\n "
            f"SOS1      {n_sos_cons}\n Nonlinear {n_nonlinear_cons}\n"
            f"Created (internal) variables:\n Binary     {n_bin_vars}\n Continuous {n_cont_vars}\n"
            f"Input Shape:  {self.input.shape}\nOutput Shape: {self.output.shape}",
            file=file,
        )

    def _create_output_vars(self, input_vars):
        """May be defined in derived class to create the output variables of predictor."""
        if (not hasattr(self, "_output") or self._output is None) and (
            not hasattr(self, "output_size") or self.output_size is None
        ):
            raise AttributeError

        if not hasattr(self, "_output") or self._output is None:
            if hasattr(self, "classification"):
                if self.classification:
                    vtype = "B"
                else:
                    vtype = "C"
            else:
                vtype = "C"
            output_vars = create_vars(
                self.scip_model,
                (input_vars.shape[0], self.output_size),
                vtype,
                lb=None,
                ub=None,
                name_prefix="out",
            )
            return output_vars
        else:
            return self._output

    @property
    def _has_solution(self):
        """Returns true if we have a solution."""
        if self.scip_model.getNSols() > 0:
            return True
        return False

    @abstractmethod
    def get_error(self, eps):
        """Returns error in SCIP's solution with respect to prediction from input.

        Returns
        -------
        error : ndarray of same shape as
            :py:attr:`pyscipopt_ml.modelling.base_predictor_constr.AbstractPredictorConstr.output`
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the absolute value of the differences between `predictor.predict(x)` and
            `y`. Where predictor is the regression / classification model represented by this object.

        Raises
        ------
        NoSolution
            If the SCIP model has no solution (either was not optimized or is infeasible).
        """
        ...

    @abstractmethod
    def _mip_model(self, **kwargs):
        """Makes MIP model for the predictor."""
        ...

    @property
    def input(self):
        """Returns the input variables of embedded predictor.

        Returns
        -------
        output : np.ndarray
        """
        return self._input

    @property
    def output(self):
        """Output variables of embedded predictor.

        Returns
        -------
        output : np.ndarray
        """
        return self._output

    @property
    def input_values(self):
        """Returns the values for the input variables if a solution is known.

        Returns
        -------
        input_vals : np.ndarray

        Raises
        ------
        NoSolution
            If SCIP has no solution (either was not optimized or is infeasible).
        """
        if not self._has_solution:
            raise NoSolution

        input_vals = np.zeros(self.input.shape)
        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[1]):
                input_vals[i][j] = self.scip_model.getVal(self.input[i][j])

        return input_vals

    @property
    def output_values(self):
        """Returns the values for the output variables if a solution is known.

        Returns
        -------
        output_value : np.ndarray

        Raises
        ------
        NoSolution
            If SCIP has no solution (either was not optimized or is infeasible).
        """
        if not self._has_solution:
            raise NoSolution

        output_vals = np.zeros(self.output.shape)
        for i in range(self.output.shape[0]):
            for j in range(self.output.shape[1]):
                output_vals[i][j] = self.scip_model.getVal(self.output[i][j])

        return output_vals
