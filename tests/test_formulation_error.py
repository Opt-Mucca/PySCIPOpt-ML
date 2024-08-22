import numpy as np
import onnx
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from pyscipopt import Model
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans, MiniBatchKMeans
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
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    Binarizer,
    Normalizer,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)
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
    if isinstance(predictor, onnx.ModelProto):
        pass
    elif not isinstance(predictor, ClusterMixin):
        if isinstance(predictor, MultiOutputClassifier) or (
            isinstance(predictor, keras.Model) and multi_dimension and classification
        ):
            y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
        predictor.fit(X, y)
    else:
        predictor.fit(X)

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
    if (
        isinstance(predictor, keras.Model) or isinstance(predictor, onnx.ModelProto)
    ) and classification:
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
    (Pipeline([("scaler", StandardScaler()), ("linreg", LinearRegression())]), True, False),
    (LinearRegression(), True, False),
    (ElasticNet(), False, False),
    (Lasso(), True, False),
    (Ridge(), False, False),
    (PLSCanonical(n_components=4), True, False),
    (PLSRegression(n_components=1), False, False),
    (Pipeline([("s", StandardScaler()), ("l", LogisticRegression())]), False, True),
    (Pipeline([("s", StandardScaler()), ("l", LogisticRegression())]), True, True),
    (Pipeline([("b", Binarizer()), ("d", DecisionTreeRegressor(max_depth=4))]), False, False),
    (DecisionTreeRegressor(max_depth=4), True, False),
    (DecisionTreeRegressor(max_depth=4), False, False),
    (DecisionTreeClassifier(max_depth=4), True, True),
    (DecisionTreeClassifier(max_depth=4), False, True),
    (GradientBoostingRegressor(max_depth=4, n_estimators=4), False, False),
    (MultiOutputRegressor(GradientBoostingRegressor(max_depth=4, n_estimators=4)), True, False),
    (GradientBoostingClassifier(max_depth=4, n_estimators=4), True, True),
    (MultiOutputClassifier(GradientBoostingClassifier(max_depth=4, n_estimators=4)), True, True),
    (RandomForestRegressor(max_depth=4, n_estimators=4), True, False),
    (RandomForestClassifier(max_depth=4, n_estimators=4), True, True),
    (RandomForestClassifier(max_depth=4, n_estimators=4), False, True),
    (onnx.load("./tests/data/sk_3_42_0_0_relu.onnx"), False, False),
    (MLPRegressor((10, 10, 10), "relu"), False, False),
    (onnx.load("./tests/data/sk_3_42_1_0_logistic.onnx"), True, False),
    (MLPRegressor((10, 10, 10), "logistic"), True, False),
    (onnx.load("./tests/data/sk_3_42_1_0_tanh.onnx"), True, False),
    (MLPRegressor((10, 10, 10), "tanh"), True, False),
    (MLPClassifier((10, 10, 10), "relu"), False, True),
    (MLPClassifier((10, 10, 10), "relu"), True, True),
    (MLPClassifier((10, 10, 10), "logistic"), True, True),
    (LinearSVR(dual="auto"), False, False),
    (MultiOutputRegressor(LinearSVR(dual="auto")), True, False),
    (SVR(kernel="poly"), False, False),
    (SVR(kernel="linear"), False, False),
    (LinearSVC(dual="auto", max_iter=5000), False, True),
    (SVC(kernel="poly", degree=2), False, True),
    (MultiOutputClassifier(SVC(kernel="poly", degree=2)), True, True),
    (XGBRegressor(max_depth=4, n_estimators=4), True, False),
    (XGBRegressor(max_depth=4, n_estimators=4), False, False),
    (XGBRFClassifier(max_depth=4, n_estimators=4), False, True),
    (LGBMRegressor(max_depth=4, n_estimators=4), False, False),
    (LGBMClassifier(max_depth=4, n_estimators=4), True, True),
    (LGBMClassifier(max_depth=4, n_estimators=4), False, True),
    (keras.Model(), True, False),
    (keras.Model(), True, True),
    (onnx.load("./tests/data/keras_3_42_1_0.onnx"), True, False),
    (onnx.load("./tests/data/keras_3_42_1_1.onnx"), True, True),
    (KMeans(n_clusters=3, n_init="auto"), True, True),
    (KMeans(n_clusters=2, n_init="auto"), False, True),
    (MiniBatchKMeans(n_clusters=3, n_init="auto"), True, True),
    (Pipeline([("s", PolynomialFeatures(degree=3)), ("l", LinearRegression())]), True, False),
    (Pipeline([("n", Normalizer(norm="l1")), ("l", LinearRegression())]), True, False),
    (Pipeline([("n", Normalizer(norm="l2")), ("l", LinearRegression())]), True, False),
    (Pipeline([("n", Normalizer(norm="max")), ("l", LinearRegression())]), True, False),
    (Pipeline([("b", Binarizer()), ("l", LinearRegression())]), True, False),
]


@pytest.mark.parametrize("predictor,multi_dimension,classification", testdata)
def test_formulation_error(predictor, multi_dimension, classification):
    train_embed_and_optimise(predictor, multi_dimension, classification)
