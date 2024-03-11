import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
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
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tensorflow import keras
from xgboost import XGBRegressor, XGBRFClassifier

from src.pyscipopt_ml import add_predictor_constr

"""
This is a set of tests for checking the basic errors of each formulation.
Single-output regression: Diabetes Dataset
Multi-output regression: Randomly generated dataset
Single-output classification: Breast Cancer Dataset
Multi-output classification: Iris Dataset
"""


def train_embed_and_optimise(predictor, multi_dimension, classification, n_samples=3, seed=42):
    # Create the training data
    if multi_dimension and classification:
        X, y = load_iris(return_X_y=True)
        input_size = 4
        output_size = 3
    elif multi_dimension and not classification:
        X, y = make_regression(
            n_samples=500, n_features=5, n_informative=5, n_targets=4, random_state=0, noise=0.5
        )
        input_size = 5
        output_size = 4
    elif not multi_dimension and classification:
        X, y = load_breast_cancer(return_X_y=True)
        input_size = 30
        output_size = 1
    else:
        X, y = load_diabetes(return_X_y=True)
        input_size = 10
        output_size = 1

    # Train the ML predictor
    if isinstance(predictor, keras.Model):
        # Define input
        predictor = keras.Sequential()
        predictor.add(keras.Input(shape=(input_size,)))

        # Add dense layers
        predictor.add(keras.layers.Dense(units=10, activation="relu"))
        predictor.add(keras.layers.Dense(units=10, activation="sigmoid"))
        predictor.add(keras.layers.Dense(units=output_size, activation="linear"))

        # Define output
        if classification and multi_dimension:
            predictor.add(keras.layers.Activation(keras.activations.softmax))
        predictor.compile(optimizer="adam", loss="mse")
    predictor.fit(X, y)

    # Create the SCIP model
    scip = Model()

    # Create the input and output variables
    input_vars = np.zeros(shape=(n_samples, input_size), dtype=object)
    output_vars = np.zeros(shape=(n_samples, output_size), dtype=object)
    # Create the random variable fixings (from samples in training)
    rng = np.random.default_rng(seed=seed)
    random_integers = rng.integers(0, len(X), size=n_samples)
    for i in range(n_samples):
        for j in range(input_size):
            input_vars[i][j] = scip.addVar(name=f"x_{i}_{j}", vtype="C")
            scip.fixVar(input_vars[i][j], X[random_integers[i]][j] + 1e-04 * rng.random())
        for j in range(output_size):
            output_vars[i][j] = scip.addVar(name=f"y_{i}_{j}", vtype="C", lb=None)

    # Embed the trained model
    if isinstance(predictor, keras.Model) and classification:
        pred_cons = add_predictor_constr(
            scip, predictor, input_vars, output_vars, output_type="classification"
        )
    else:
        pred_cons = add_predictor_constr(scip, predictor, input_vars, output_vars)

    # Optimise the model and check the error
    print(f"Optimising predictor {predictor}")
    # scip.hideOutput()
    scip.optimize()

    # Ensure the error is below some threshold
    if np.max(pred_cons.get_error()) > 10**-4:
        error = np.max(pred_cons.get_error())
        raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -4}")


testdata = [
    (LinearRegression(), True, False),
    (ElasticNet(), False, False),
    (Lasso(), True, False),
    (Ridge(), False, False),
    (PLSCanonical(n_components=4), True, False),
    (PLSRegression(n_components=1), False, False),
    (LogisticRegression(), False, True),
    (LogisticRegression(), True, True),
    (DecisionTreeRegressor(max_depth=4), True, False),
    (DecisionTreeRegressor(max_depth=4), False, False),
    (DecisionTreeClassifier(max_depth=4), True, True),
    (DecisionTreeClassifier(max_depth=4), False, True),
    (GradientBoostingRegressor(max_depth=4, n_estimators=4), False, False),
    (GradientBoostingClassifier(max_depth=4, n_estimators=4), True, True),
    (RandomForestRegressor(max_depth=4, n_estimators=4), True, False),
    (RandomForestClassifier(max_depth=4, n_estimators=4), True, True),
    (RandomForestClassifier(max_depth=4, n_estimators=4), False, True),
    (MLPRegressor(hidden_layer_sizes=(10, 10, 10), activation="relu"), False, False),
    (MLPRegressor(hidden_layer_sizes=(10, 10, 10), activation="logistic"), True, False),
    (MLPRegressor(hidden_layer_sizes=(10, 10, 10), activation="tanh"), True, False),
    (MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation="relu"), False, True),
    (MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation="relu"), True, True),
    (MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation="logistic"), True, True),
    (LinearSVR(), False, False),
    (SVR(kernel="poly"), False, False),
    (SVR(kernel="linear"), False, False),
    (LinearSVC(), False, True),
    (SVC(kernel="poly", degree=2), False, True),
    (XGBRegressor(max_depth=4, n_estimators=4), True, False),
    (XGBRegressor(max_depth=4, n_estimators=4), False, False),
    (XGBRFClassifier(max_depth=4, n_estimators=4), False, True),
    (LGBMRegressor(max_depth=4, n_estimators=4), False, False),
    (LGBMClassifier(max_depth=4, n_estimators=4), True, True),
    (LGBMClassifier(max_depth=4, n_estimators=4), False, True),
    (keras.Model(), True, False),
    (keras.Model(), True, True),
]


@pytest.mark.parametrize("predictor,multi_dimension,classification", testdata)
def test_formulation_error(predictor, multi_dimension, classification):
    train_embed_and_optimise(predictor, multi_dimension, classification)
