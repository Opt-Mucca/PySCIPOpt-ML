import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyscipopt import Model
from sklearn.neural_network import MLPRegressor
from tensorflow import keras
from torch.utils.data import DataLoader, TensorDataset

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
xQ_1x + A_1x + c_1
xQ_2x + A_2x + c_2

Let f be a NN that approximates the first quadratic function, and g be a NN
that approximates the second quadratic function.
Let const be some constant value in the range of the randomly generated quadratic function

The MIP model is:

g(x) = const
min(f(x))
"""


def build_and_optimise_function_approximation_model(
    seed=42,
    n_inputs=5,
    n_samples=1000,
    framework="sklearn",
    layers_sizes=(20, 20, 10),
    build_only=False,
):
    assert len(layers_sizes) == 3

    X, y_1, y_2 = build_random_quadratic_functions(
        seed=seed, n_inputs=n_inputs, n_samples=n_samples
    )

    if framework == "sklearn":
        reg_1 = MLPRegressor(
            random_state=seed,
            hidden_layer_sizes=(layers_sizes[0], layers_sizes[1], layers_sizes[2]),
        ).fit(X, y_1.reshape(-1))
        reg_2 = MLPRegressor(
            random_state=seed,
            hidden_layer_sizes=(layers_sizes[0], layers_sizes[1], layers_sizes[2]),
        ).fit(X, y_2.reshape(-1))
    elif framework == "keras":
        keras.utils.set_random_seed(seed)
        reg_1 = keras.Sequential()
        reg_1.add(keras.Input(shape=(n_inputs,)))
        reg_1.add(keras.layers.Dense(layers_sizes[0], activation="linear"))
        reg_1.add(keras.layers.Activation(keras.activations.relu))
        reg_1.add(keras.layers.Dense(layers_sizes[1], activation="sigmoid"))
        reg_1.add(keras.layers.Dense(layers_sizes[2], activation="relu"))
        reg_1.add(keras.layers.Dense(1, activation="linear"))
        reg_1.compile(optimizer="adam", loss="mse")
        reg_1.fit(X, y_1, batch_size=32, epochs=50)
        reg_2 = keras.Sequential()
        reg_2.add(keras.Input(shape=(n_inputs,)))
        reg_2.add(keras.layers.Dense(layers_sizes[0], activation="linear"))
        reg_2.add(keras.layers.Activation(keras.activations.sigmoid))
        reg_2.add(keras.layers.Dense(layers_sizes[1], activation="tanh"))
        reg_2.add(keras.layers.Dense(layers_sizes[2], activation="relu"))
        reg_2.add(keras.layers.Dense(1, activation="linear"))
        reg_2.compile(optimizer="adam", loss="mse")
        reg_2.fit(X, y_2, batch_size=32, epochs=200)
    elif framework == "torch":
        torch.random.manual_seed(seed)
        reg_1 = nn.Sequential(
            nn.Linear(n_inputs, layers_sizes[0]),
            nn.ReLU(),
            nn.Linear(layers_sizes[0], layers_sizes[1]),
            nn.ReLU(),
            nn.Linear(layers_sizes[1], layers_sizes[2]),
            nn.ReLU(),
            nn.Linear(layers_sizes[2], 1),
        )
        reg_2 = nn.Sequential(
            nn.Linear(n_inputs, layers_sizes[0]),
            nn.ReLU(),
            nn.Linear(layers_sizes[0], layers_sizes[1]),
            nn.ReLU(),
            nn.Linear(layers_sizes[1], layers_sizes[2]),
            nn.ReLU(),
            nn.Linear(layers_sizes[2], 1),
        )

        # Convert data into PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_1_tensor = torch.tensor(y_1, dtype=torch.float32)
        y_2_tensor = torch.tensor(y_2, dtype=torch.float32)

        # Create a DataLoader for handling batches
        dataset_1 = TensorDataset(X_tensor, y_1_tensor)
        dataset_2 = TensorDataset(X_tensor, y_2_tensor)
        batch_size = 32
        dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, shuffle=True)
        dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, shuffle=True)

        # Initialise the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer_1 = optim.Adam(reg_1.parameters(), lr=0.001, weight_decay=0.0001)
        optimizer_2 = optim.Adam(reg_2.parameters(), lr=0.001, weight_decay=0.0001)

        # Training loop
        for epoch in range(200):
            for batch_X, batch_y in dataloader_1:
                # Forward pass
                outputs = reg_1(batch_X)

                # Calculate loss
                loss = criterion(outputs, batch_y.view(-1, 1))  # Assuming y is a 1D array

                # Backward pass and optimization
                optimizer_1.zero_grad()
                loss.backward()
                optimizer_1.step()
            for batch_X, batch_y in dataloader_2:
                # Forward pass
                outputs = reg_2(batch_X)

                # Calculate loss
                loss = criterion(outputs, batch_y.view(-1, 1))  # Assuming y is a 1D array

                # Backward pass and optimization
                optimizer_2.zero_grad()
                loss.backward()
                optimizer_2.step()
    else:
        raise ValueError(f"Framework {framework} is unknown")

    # Now build the SCIP Model and embed the neural networks
    scip, input_vars, output_vars = build_basic_scip_model(n_inputs)
    mlp_cons_1 = add_predictor_constr(
        scip, reg_1, input_vars[0], output_vars[0], unique_naming_prefix="reg_1_"
    )
    mlp_cons_2 = add_predictor_constr(
        scip, reg_2, input_vars[1], output_vars[1], unique_naming_prefix="reg_2_"
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
        np.random.uniform(0, 5, size=(2, n_inputs, n_inputs)), decimals=3
    )
    linear_coefficients = np.round(np.random.uniform(0, 1, size=(2, n_inputs)), decimals=3)
    constant = np.round(np.random.uniform(0, 1, size=(2,)), decimals=3)

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


def build_basic_scip_model(n_inputs):
    # Initialise a SCIP Model
    scip = Model()

    # Create the input variables
    input_vars = np.zeros((2, n_inputs), dtype=object)
    for i in range(2):
        for j in range(n_inputs):
            # Tight bounds are important for MIP formulations of neural networks. They often drastically improve
            # performance. As our training data is in the range [-10, 10], we pass that as bounds [-10, 10].
            # These bounds will then propagate to other variables.
            input_vars[i][j] = scip.addVar(name=f"x_{i}_{j}", vtype="C", lb=-10, ub=10)

    # Create the output variables. (Note that these variables will be automatically constructed if not specified)
    output_vars = np.zeros((2, 1), dtype=object)
    for i in range(2):
        output_vars[i] = scip.addVar(name=f"y_{i}", vtype="C", lb=None, ub=None)

    # Now set additional constraints and set the objective
    scip.addCons(output_vars[1][0] == 10, name="fix_output_reg_2")
    scip.setObjective(output_vars[0][0] + 10000)

    return scip, input_vars, output_vars


def test_sklearn_mlp_regression():
    scip = build_and_optimise_function_approximation_model(
        seed=42, n_inputs=5, n_samples=1000, framework="sklearn", layers_sizes=(20, 20, 10)
    )


def test_torch_sequential_regression():
    scip = build_and_optimise_function_approximation_model(
        seed=42, n_inputs=5, n_samples=1000, framework="torch", layers_sizes=(20, 20, 10)
    )


def test_keras_sequential_regression():
    scip = build_and_optimise_function_approximation_model(
        seed=42, n_inputs=5, n_samples=1000, framework="keras", layers_sizes=(10, 10, 10)
    )
