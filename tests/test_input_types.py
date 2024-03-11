import numpy as np
import pytest
from pyscipopt import Model
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_regression,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.pyscipopt_ml import add_predictor_constr

"""
Tests for the various possible input types of each ML Model from sklearn.
load_iris is used for multi-classification, load_diabetes used for regression,
load_breast_cancer used for single class classification, and laod_wine is used for
multiple output regression.

For all models tests are run on input types being lists and numpy arrays of various dimensions.
Tests are additionally run with and without the output defined by the user.
"""


def get_data_and_create_basic_model(
    predictor,
    multi_dimension=False,
    classification=False,
    n_samples=1,
    input_type="numpy",
    output_type="numpy",
):
    assert input_type in ("numpy", "list")
    assert output_type in ("numpy", "list", "None")

    # Load the correct data
    if multi_dimension and classification:
        X, y = load_iris(return_X_y=True)
        input_size = 4
        output_size = 3
    elif multi_dimension and not classification:
        X, y = make_regression(
            n_samples=500, n_features=10, n_informative=5, n_targets=2, random_state=0, noise=0.5
        )
        input_size = 10
        output_size = 2
    elif not multi_dimension and classification:
        X, y = load_breast_cancer(return_X_y=True)
        input_size = 30
        output_size = 1
    else:
        X, y = load_diabetes(return_X_y=True)
        input_size = 10
        output_size = 1

    # Fit the predictor
    predictor = predictor().fit(X, y)

    # Create the SCIP model
    scip = Model()

    # Create the input variables
    if input_type == "numpy":
        input_vars = np.zeros(shape=(n_samples, input_size), dtype=object)
    else:
        input_vars = [[None for _ in range(input_size)] for _ in range(n_samples)]
    for i in range(n_samples):
        for j in range(input_size):
            input_vars[i][j] = scip.addVar(name=f"x_{i}", vtype="C", lb=-10, ub=10)

    # Create the output variables
    if output_type == "numpy":
        output_vars = np.zeros(shape=(n_samples, output_size), dtype=object)
    elif output_type == "list":
        output_vars = [[None for _ in range(output_size)] for _ in range(n_samples)]
    else:
        output_vars = None
    if output_type != "None":
        for i in range(n_samples):
            for j in range(output_size):
                output_vars[i][j] = scip.addVar(name=f"y_{i}", vtype="C", lb=-10, ub=10)

    pred_cons = add_predictor_constr(scip, predictor, input_vars, output_vars)


testdata = [
    (LinearRegression, True, False, 2, "numpy", "numpy"),
    (ElasticNet, False, False, 1, "numpy", "numpy"),
    (LinearRegression, False, False, 2, "numpy", "numpy"),
    (Lasso, True, False, 1, "list", "numpy"),
    (LinearRegression, True, False, 2, "numpy", "list"),
    (Ridge, True, False, 2, "numpy", "None"),
    (LogisticRegression, True, True, 2, "numpy", "numpy"),
    (LogisticRegression, False, True, 1, "numpy", "numpy"),
    (LinearRegression, True, False, 2, "list", "None"),
    (PLSRegression, True, False, 1, "numpy", "numpy"),
    (PLSCanonical, True, False, 2, "numpy", "None"),
    (DecisionTreeRegressor, True, False, 2, "numpy", "numpy"),
    (DecisionTreeRegressor, False, False, 1, "numpy", "numpy"),
    (DecisionTreeRegressor, True, False, 2, "list", "None"),
    (DecisionTreeClassifier, True, True, 2, "numpy", "numpy"),
    (DecisionTreeClassifier, False, True, 2, "list", "None"),
    (GradientBoostingRegressor, False, False, 2, "numpy", "numpy"),
    (GradientBoostingRegressor, False, False, 2, "list", "None"),
    (GradientBoostingClassifier, True, True, 2, "numpy", "numpy"),
    (GradientBoostingClassifier, False, True, 2, "list", "None"),
    (RandomForestRegressor, True, False, 2, "numpy", "numpy"),
    (RandomForestRegressor, False, False, 1, "numpy", "numpy"),
    (RandomForestRegressor, False, False, 2, "list", "None"),
    (RandomForestClassifier, False, True, 1, "numpy", "numpy"),
    (RandomForestClassifier, False, True, 2, "list", "None"),
    (MLPRegressor, True, False, 1, "numpy", "numpy"),
    (MLPRegressor, False, False, 2, "list", "None"),
    (MLPClassifier, True, True, 1, "numpy", "numpy"),
    (MLPClassifier, False, True, 2, "list", "None"),
    (LinearSVR, False, False, 2, "list", "None"),
    (LinearSVC, False, True, 1, "numpy", "numpy"),
]


@pytest.mark.parametrize(
    "predictor,multi_dimension,classification,n_samples,input_type,output_type", testdata
)
def test_input_output(
    predictor, multi_dimension, classification, n_samples, input_type, output_type
):
    get_data_and_create_basic_model(
        predictor, multi_dimension, classification, n_samples, input_type, output_type
    )
