import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyscipopt import Model, quicksum
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we take the point of view someone trying to fool a trained predictor for a given example.
This is referred to in the literature as an adversarial example.

We have access to open source data (thanks to: @article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  volume={86},
  number={11},
  pages={2278--2324},
  year={1998},
  publisher={Ieee}
})

This data is the famous MNIST dataset, which contains images of individual numbers that have been
drawn by children. Each image contains exactly one centred image, and is available everywhere as the
standard deep learning data set.

The goal of this MIP is to take an image from the dataset, and perturb it such that
the predictor now misclassifies it. There is a budget on how much the image can be perturbed.

Let I be the x coordinate and J be the y coordinate of the image.
Let f be the ML predictor that outputs the probability of each label for the input image
Let image[i][j] be the input image in grey scale with pixel values between 0-1.
Let x[i][j] the perturbed image in grey scale with pixel values between 0-1.
Let y be the probability of each label given the ML predictor
W.l.o.g assume that y[0] is the true label and y[1] is the fake label.

x[i][j] - image[i][j] <= abs_diff[i][j] for all i, j
image[i][j] - x[i][j] <= abs_diff[i][j] for all i, j
\sum_i,j abs_diff[i][j] <= 5
y = f(x)

max(y[1] - y[0])
"""


def build_and_optimise_adversarial_mnist_torch(
    data_seed=42,
    training_seed=42,
    n_pixel_1d=16,
    layer_size=16,
    n_layers=2,
    test=True,
    formulation="sos",
    build_only=False,
):
    # Set random seed for reproducibility and select the image that is going to be perturbed
    data_random_state = np.random.RandomState(data_seed)
    image_number = data_random_state.randint(low=0, high=30000)
    torch.manual_seed(training_seed)

    # Define transformations for the MNIST dataset
    transform = transforms.Compose(
        [
            transforms.Resize(n_pixel_1d),  # Resize the image
            transforms.ToTensor(),  # Convert images to tensors. This automatically normalises to [0,1]
        ]
    )

    # Get MNIST digit recognition data set
    train_dataset = datasets.MNIST(
        root="./tests/data/MNIST", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./tests/data/MNIST", train=False, transform=transform, download=True
    )

    # Create DataLoader for handling batches
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create the neural network
    layers = [nn.Flatten(), nn.Linear(n_pixel_1d**2, layer_size), nn.ReLU()]
    nn.init.xavier_uniform_(layers[1].weight)
    for i in range(n_layers - 1):
        layers.append(nn.Linear(layer_size, layer_size))
        nn.init.xavier_uniform_(layers[-1].weight)
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_size, 10))
    nn.init.xavier_uniform_(layers[-1].weight)
    reg = nn.Sequential(*layers)
    nn.utils.clip_grad_norm_(reg.parameters(), 0.1)

    # If the model is already saved then skip the training step
    saved_neural_network_path = "./tests/data/adversarial.pt"
    if test and os.path.isfile(saved_neural_network_path):
        reg.load_state_dict(torch.load(saved_neural_network_path))
        reg.eval()

    else:
        # Initialise the loss function, and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(reg.parameters(), weight_decay=0.001)

        # Training loop
        num_epochs = 20

        for epoch in range(num_epochs):
            reg.train()
            for batch_X, batch_y in train_dataloader:
                # Forward pass
                outputs = reg(batch_X)

                # Calculate loss
                loss = criterion(outputs, batch_y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print training loss after each epoch
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}")

        # Test the trained model on the test dataset
        reg.eval()
        correct = 0
        total = 0

        # Determine the accuracy of the trained model on the test dataset
        with torch.no_grad():
            for batch_X, batch_y in test_dataloader:
                outputs = reg(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Save the neural network for repeated tests
        if test:
            torch.save(reg.state_dict(), saved_neural_network_path)

    # Now create the MILP Model. The MILP will have a budget of pixels to change and decide which pixels and which new
    # values most change the classification of an image
    scip = Model()

    output_values = reg.forward(train_dataset[image_number][0])
    sorted_labels = torch.argsort(output_values)
    right_label = sorted_labels[0][-1]
    wrong_label = sorted_labels[0][-2]

    # Create the input variables
    input_vars = np.zeros((1, n_pixel_1d, n_pixel_1d), dtype=object)
    for i in range(n_pixel_1d):
        for j in range(n_pixel_1d):
            # Tight bounds are important for MIP formulations of neural networks. They often drastically improve
            # performance. As our data is a scaled image, it is in the range [0, 1].
            # These bounds will then propagate to other variables.
            input_vars[0][i][j] = scip.addVar(name=f"x_{i}_{j}", vtype="C", lb=0, ub=1)

    # Create the output variables. (Note that these variables will be automatically constructed if not specified)
    output_vars = np.zeros((10,), dtype=object)
    for i in range(10):
        output_vars[i] = scip.addVar(name=f"y_{i}", vtype="C", lb=None, ub=None)

    # Create the difference variables
    sum_max_diff = data_random_state.uniform(low=4.5, high=5.5)
    abs_diff = np.zeros((n_pixel_1d, n_pixel_1d), dtype=object)
    for i in range(n_pixel_1d):
        for j in range(n_pixel_1d):
            abs_diff[i][j] = scip.addVar(name=f"abs_diff_{i}", vtype="C", lb=0, ub=1)

    # Create constraints ensuring only a total certain amount of the picture can change
    for i in range(n_pixel_1d):
        for j in range(n_pixel_1d):
            scip.addCons(
                abs_diff[i][j] >= input_vars[0][i][j] - train_dataset[image_number][0][0][i][j]
            )
            scip.addCons(
                abs_diff[i][j] >= -input_vars[0][i][j] + train_dataset[image_number][0][0][i][j]
            )

    scip.addCons(
        quicksum(quicksum(abs_diff[i][j] for j in range(n_pixel_1d)) for i in range(n_pixel_1d))
        <= sum_max_diff
    )

    # Set an objective to maximise the difference between the correct and the wrong label
    scip.setObjective(-output_vars[wrong_label] + output_vars[right_label] + 1)

    # Add the ML constraint
    pred_cons = add_predictor_constr(
        scip,
        reg,
        input_vars,
        output_vars,
        unique_naming_prefix="adversarial_",
        formulation=formulation,
    )

    if not build_only:
        scip.optimize()

        # We can check the "error" of the MIP embedding by determining the difference between the Torch and SCIP output
        if np.max(pred_cons.get_error()) > 10**-3:
            error = np.max(pred_cons.get_error())
            raise AssertionError(f"Max error {error} exceeds threshold of {10**-3}")

    return scip


def test_mnist_torch():
    scip = build_and_optimise_adversarial_mnist_torch(
        data_seed=42,
        training_seed=42,
        n_pixel_1d=12,
        layer_size=10,
        n_layers=2,
        test=True,
        formulation="sos",
    )


def test_mnist_torch_bigm():
    scip = build_and_optimise_adversarial_mnist_torch(
        data_seed=42,
        training_seed=42,
        n_pixel_1d=12,
        layer_size=10,
        n_layers=2,
        test=True,
        formulation="bigm",
    )
