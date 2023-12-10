import csv


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
