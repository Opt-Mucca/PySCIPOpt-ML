"""Exceptions for pyscipopt-ml."""


class NotRegistered(Exception):
    """Predictor is not supported by pyscipopt-ml."""

    def __init__(self, predictor):
        super().__init__(
            f"Object of type {predictor} is not registered/supported with pyscipopt-ml"
        )


class NoModel(Exception):
    """No model is known for some structure."""

    def __init__(self, predictor, reason):
        if not isinstance(predictor, str):
            predictor = type(predictor).__name__
        super().__init__(f"Can't do model for {predictor}: {reason}")


class NoSolution(Exception):
    """SCIP doesn't have a solution."""

    def __init__(self):
        super().__init__("No solution available")


class ParameterError(Exception):
    """Wrong parameter to a function."""

    def __init__(self, message):
        super().__init__(message)
