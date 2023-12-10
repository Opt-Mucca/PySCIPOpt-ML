Basic Example - Function Approximation
######################################

Explanation and Description
===========================

In this tutorial, we will see how to embed a trained ML predictor into
a SCIP Model.

The scenario for this basic example is that we have two non-linear functions,
which we will approximate by some ML technique (in this case a neural network).
The aim of the optimisation problem is to maximise the approximation of the first function,
while satisfying some equality constraint on the second. We can describe this scenario mathematically.

Let :math:`\mathbf{x}` be the input to both functions. Let :math:`f(\mathbf{x})` be the first function, and
:math:`g(\mathbf{x})` be the second function. Their approximations via some ML technique are
:math:`f'(\mathbf{x})` and :math:`g'(\mathbf{x})`. The MIP is:

.. math::

    \begin{align*}
    &\text{min    }& f'(\mathbf{x}) \\
    &\text{s.t.}   & g'(\mathbf{x}) = 10
    \end{align*}


Code Walkthrough
=================

To begin with, we first need to be able to generate the non-linear functions:

.. code-block:: python

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


Now that we have the ability to create training data for two non-linear functions,
let us make a general function for creating the SCIP model that we will embed the
trained ML predictors.

.. code-block:: python

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
        scip.setObjective(output_vars[0][0])

        return scip, input_vars, output_vars


We now are only lacking the trained ML predictor to insert into the SCIP model. So we now train a ML predictor
In the example below, we provide a function that has the ability to train either a Scikit-Learn MLPRegressor,
or a PyTorch fully connected Sequential model with ReLU activation functions.

.. code-block:: python

    def build_and_optimise_function_approximation_model(
    seed=42, n_inputs=5, n_samples=1000, sklearn_or_torch="sklearn", layers_sizes=(20, 20, 10)
    ):
        assert len(layers_sizes) == 3

        X, y_1, y_2 = build_random_quadratic_functions(
            seed=seed, n_inputs=n_inputs, n_samples=n_samples
        )

        if sklearn_or_torch == "sklearn":
            reg_1 = MLPRegressor(
                random_state=seed,
                hidden_layer_sizes=(layers_sizes[0], layers_sizes[1], layers_sizes[2]),
            ).fit(X, y_1.reshape(-1))
            reg_2 = MLPRegressor(
                random_state=seed,
                hidden_layer_sizes=(layers_sizes[0], layers_sizes[1], layers_sizes[2]),
            ).fit(X, y_2.reshape(-1))
        else:
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

        # Now build the SCIP Model and embed the neural networks
        scip, input_vars, output_vars = build_basic_scip_model(n_inputs)
        mlp_cons_1 = add_predictor_constr(
            scip, reg_1, input_vars[0], output_vars[0], unique_naming_prefix="reg_1_"
        )
        mlp_cons_2 = add_predictor_constr(
            scip, reg_2, input_vars[1], output_vars[1], unique_naming_prefix="reg_2_"
        )

        return scip



To execute the above code we can now run:

.. code-block:: python

    # Get the SCIP Model with the embedded trained predictors
    scip = build_and_optimise_function_approximation_model()

    # Optimize the model
    scip.optimize()

    # We can check the "error" of the MIP embedding via the difference between SKLearn / Torch and SCIP output
    if np.max(mlp_cons_1.get_error()) > 10**-3:
        error = np.max(mlp_cons_1.get_error())
        raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -3}")
    if np.max(mlp_cons_2.get_error()) > 10**-3:
        error = np.max(mlp_cons_2.get_error())
        raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -3}")