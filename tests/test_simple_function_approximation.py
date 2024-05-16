import numpy as np
from pyscipopt import Model
from sklearn.neural_network import MLPRegressor
from tensorflow import keras
from utils import train_torch_neural_network

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we are performing simple function approximations.
There are two non-linear functions that have been approximated by neural networks.
We want to minimise the value of the first function, while satisfying some equality of the second.

This is a simple example to test the performance of the MIP formulations for NNs
with ReLU activation functions:
@article{grimstad2019relu,
  title={ReLU networks as surrogate models in mixed-integer linear programs},
  author={Grimstad, Bjarne and Andersson, Henrik},
  journal={Computers and Chemical Engineering},
  volume={131},
  pages={106580},
  year={2019},
  publisher={Elsevier}
}

Let there be two random quadratic functions:
xQ_1x + a_1x + c_1
xQ_2x + a_2x + c_2

Let f be a NN that approximates the first quadratic function, and g be a NN
that approximates the second quadratic function.
Let const be some constant value in the range of the randomly generated quadratic function

The MIP model is:

g(x) = const
min(f(x))
"""


def build_and_optimise_function_approximation_model(
    n_inputs=5,
    n_samples=2500,
    framework="sklearn",
    formulation="sos",
    layer_size=8,
    n_layers=3,
    training_seed=42,
    data_seed=42,
    build_only=False,
):
    X, y_1, y_2 = build_random_quadratic_functions(
        seed=data_seed, n_inputs=n_inputs, n_samples=n_samples
    )

    if framework == "sklearn":
        hidden_layer_sizes = tuple([layer_size for _ in range(n_layers)])
        reg_1 = MLPRegressor(
            random_state=training_seed,
            hidden_layer_sizes=hidden_layer_sizes,
        ).fit(X, y_1.reshape(-1))
        reg_2 = MLPRegressor(
            random_state=training_seed,
            hidden_layer_sizes=hidden_layer_sizes,
        ).fit(X, y_2.reshape(-1))
    elif framework == "keras":
        keras.utils.set_random_seed(training_seed)
        reg_1 = keras.Sequential()
        reg_2 = keras.Sequential()
        reg_1.add(keras.Input(shape=(n_inputs,)))
        reg_2.add(keras.Input(shape=(n_inputs,)))
        for _ in range(n_layers):
            reg_1.add(keras.layers.Dense(layer_size, activation="relu"))
            reg_2.add(keras.layers.Dense(layer_size, activation="relu"))
        reg_1.add(keras.layers.Dense(1, activation="linear"))
        reg_2.add(keras.layers.Dense(1, activation="linear"))
        reg_1.compile(optimizer="adam", loss="mse")
        reg_1.fit(X, y_1, batch_size=64, epochs=20)
        reg_2.compile(optimizer="adam", loss="mse")
        reg_2.fit(X, y_2, batch_size=64, epochs=20)
    elif framework == "torch":
        reg_1 = train_torch_neural_network(
            X, y_1, n_layers, layer_size, training_seed, reshape=True
        )
        reg_2 = train_torch_neural_network(
            X, y_2, n_layers, layer_size, training_seed, reshape=True
        )
    else:
        raise ValueError(f"Framework {framework} is unknown")

    # Now build the SCIP Model and embed the neural networks
    scip, input_vars, output_vars = build_basic_scip_model(n_inputs, np.median(y_2))
    mlp_cons_1 = add_predictor_constr(
        scip,
        reg_1,
        input_vars,
        output_vars[0],
        unique_naming_prefix="reg_1_",
        formulation=formulation,
    )
    mlp_cons_2 = add_predictor_constr(
        scip,
        reg_2,
        input_vars,
        output_vars[1],
        unique_naming_prefix="reg_2_",
        formulation=formulation,
    )

    if not build_only:
        # Optimize the model
        scip.optimize()

        # We can check the "error" of the MIP embedding via the difference between SKLearn / Torch and SCIP output
        if np.max(mlp_cons_1.get_error()) > 2 * 10**-3:
            error = np.max(mlp_cons_1.get_error())
            raise AssertionError(f"Max error {error} exceeds threshold of {2 * 10 ** -3}")
        if np.max(mlp_cons_2.get_error()) > 2 * 10**-3:
            error = np.max(mlp_cons_2.get_error())
            raise AssertionError(f"Max error {error} exceeds threshold of {2 * 10 ** -3}")

    return scip


def build_random_quadratic_functions(seed=42, n_inputs=5, n_samples=1000):
    # Set the random seed so the results don't change
    np.random.seed(seed)

    # Generate two random quadratic functions f(x) = x^{t}Qx + Ax + c
    quadratic_coefficients = np.round(
        np.random.uniform(-5, 5, size=(2, n_inputs, n_inputs)), decimals=3
    )
    linear_coefficients = np.round(np.random.uniform(-1, 1, size=(2, n_inputs)), decimals=3)
    constant = np.round(np.random.uniform(-1, 1, size=(2,)), decimals=3)

    # Generate data from the quadratic function
    X = np.random.uniform(-10, 10, size=(n_samples, n_inputs))
    y_1 = np.zeros((n_samples,))
    y_2 = np.zeros((n_samples,))
    for i in range(n_samples):
        y_1[i] = (
            X[i] @ quadratic_coefficients[0] @ X[i].T
            + linear_coefficients[0] @ X[i].T
            + constant[0]
        )
        y_2[i] = (
            X[i] @ quadratic_coefficients[1] @ X[i].T
            + linear_coefficients[1] @ X[i].T
            + constant[1]
        )

    return X, y_1, y_2


def build_basic_scip_model(n_inputs, intercept):
    # Initialise a SCIP Model
    scip = Model()

    # Create the input variables
    input_vars = np.zeros((1, n_inputs), dtype=object)
    for i in range(n_inputs):
        # Tight bounds are important for MIP formulations of neural networks. They often drastically improve
        # performance. As our training data is in the range [-10, 10], we pass a multiple as bounds [-20, 20].
        # These bounds will then propagate to other variables.
        input_vars[0][i] = scip.addVar(name=f"x_{i}", vtype="C", lb=-20, ub=20)

    # Create the output variables. (Note that these variables will be automatically constructed if not specified)
    output_vars = np.zeros((2, 1), dtype=object)
    for i in range(2):
        output_vars[i] = scip.addVar(name=f"y_{i}", vtype="C", lb=None, ub=None)

    # Now set additional constraints and set the objective
    scip.addCons(output_vars[1][0] == intercept, name="fix_output_reg_2")
    scip.setObjective(output_vars[0][0] + 10000)

    return scip, input_vars, output_vars


def test_sklearn_mlp_regression():
    scip = build_and_optimise_function_approximation_model(
        data_seed=42,
        training_seed=42,
        n_inputs=5,
        n_samples=1000,
        framework="sklearn",
        formulation="sos",
        layer_size=8,
    )


def test_sklearn_mlp_regression_bigm():
    scip = build_and_optimise_function_approximation_model(
        data_seed=42,
        training_seed=42,
        n_inputs=5,
        n_samples=1000,
        framework="sklearn",
        formulation="bigm",
        layer_size=8,
    )


def test_torch_sequential_regression():
    scip = build_and_optimise_function_approximation_model(
        data_seed=42,
        training_seed=42,
        n_inputs=5,
        n_samples=1000,
        framework="torch",
        formulation="sos",
        layer_size=8,
    )


def test_torch_sequential_regression_bigm():
    scip = build_and_optimise_function_approximation_model(
        data_seed=42,
        training_seed=42,
        n_inputs=5,
        n_samples=1000,
        framework="torch",
        formulation="bigm",
        layer_size=8,
    )


def test_keras_sequential_regression():
    scip = build_and_optimise_function_approximation_model(
        data_seed=42,
        training_seed=42,
        n_inputs=5,
        n_samples=1000,
        framework="keras",
        formulation="sos",
        layer_size=3,
    )


def test_keras_sequential_regression_bigm():
    scip = build_and_optimise_function_approximation_model(
        data_seed=42,
        training_seed=42,
        n_inputs=5,
        n_samples=1000,
        framework="keras",
        formulation="bigm",
        layer_size=3,
    )
