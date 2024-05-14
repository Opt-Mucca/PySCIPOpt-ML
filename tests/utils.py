import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def read_csv_to_dict(csv_file_path):
    """
    Function for reading a csv file in the traditional Kaggle style.

    Parameters
    ----------
    csv_file_path : filepath
        The file path to the csv file. Traditionally in ./tests/data/

    Returns
    -------
    data_dict : dict
        Dictionary containing the csv file data. Keys to the dictionary are the headers of the csv.
    """

    # Empty dictionary containing loaded data
    data_dict = {}

    # Open the CSV file
    with open(csv_file_path, "r") as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)

        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Iterate through each column in the row
            for column, value in row.items():
                # Check if the column is already a key in the dictionary
                if column in data_dict:
                    # Append the value to the existing list
                    data_dict[column].append(value)
                else:
                    # If the column is not a key, create a new list with the value
                    data_dict[column] = [value]

    return data_dict


def train_torch_neural_network(
    input_data,
    labels,
    num_layers,
    layer_size,
    training_seed,
    reshape=False,
    outdim=1,
    binary_classifier=False,
):
    torch.manual_seed(training_seed)
    layers = [nn.Linear(input_data.shape[1], layer_size), nn.ReLU()]
    nn.init.xavier_uniform_(layers[0].weight)
    for i in range(num_layers - 1):
        layers.append(nn.Linear(layer_size, layer_size))
        nn.init.xavier_uniform_(layers[-1].weight)
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_size, outdim))
    nn.init.xavier_uniform_(layers[-1].weight)
    if binary_classifier:
        layers.append(nn.Sigmoid())
    reg = nn.Sequential(*layers)
    nn.utils.clip_grad_norm_(reg.parameters(), 0.1)
    if binary_classifier:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(reg.parameters(), weight_decay=0.001)
    reg.train()
    if reshape:
        dataloader = DataLoader(
            TensorDataset(torch.Tensor(input_data), torch.Tensor(labels.reshape(-1, 1))),
            batch_size=64,
            shuffle=True,
        )
    else:
        dataloader = DataLoader(
            TensorDataset(torch.Tensor(input_data), torch.Tensor(labels)),
            batch_size=64,
            shuffle=True,
        )
    for epoch in range(20):
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = reg(batch_X)
            # Calculate loss
            loss = criterion(outputs, batch_y)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print training loss after each epoch
        print(f"Epoch [{epoch + 1}/{20}], Training Loss: {loss.item():.4f}")

    return reg
