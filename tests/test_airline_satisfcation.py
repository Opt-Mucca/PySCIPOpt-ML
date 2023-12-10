import numpy as np
from pyscipopt import Model
from sklearn.neural_network import MLPClassifier
from utils import read_csv_to_dict

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we take the point of view of a conglomeration of airlines.
This set of airlines is worried about current customer satisfaction levels,
and has assigned a budget to improve the predicted levels. Each airline, however,
has its own customer base, and incentives for which bit of the flight experience to
improve. Using the budget, maximise the total improved customer satisfaction levels.
"""

# Path to water potability data
data_dict = read_csv_to_dict("./tests/data/airline_satisfaction.csv")

# The features of our predictor. All distance based features are variables.
features = [
    ("Gender", "Male"),
    ("Customer Type", "Loyal"),
    ("Type of Travel", "Business travel"),
    ("Class", "Eco"),
    ("Flight Distance", int),
    ("Inflight wifi service", int),
    ("Departure/Arrival time convenient", int),
    ("Ease of Online booking", int),
    ("Gate location", int),
    ("Food and drink", int),
    ("Online boarding", int),
    ("Seat comfort", int),
    ("Inflight entertainment", int),
    ("On-board service", int),
    ("Leg room service", int),
    ("Baggage handling", int),
    ("Checkin service", int),
    ("Inflight service", int),
    ("Cleanliness", int),
]
feature_to_idx = {feature: i for i, feature in enumerate(features)}
n_features = len(features)
n_passengers = 200
np.random.seed(42)

# Generate the actual input data arrays for the ML predictors
X = []
y = np.array([x == "satisfied" for x in data_dict["satisfaction"]]).astype(int).reshape(-1, 1)
for feature, feature_key in features:
    if feature == "Class":
        X.append(np.array([("Eco" not in x) for x in data_dict[feature]]).astype(int))
    elif feature_key != int:
        X.append(np.array([x == feature_key for x in data_dict[feature]]).astype(int))
    else:
        X.append(np.array([int(x) for x in data_dict[feature]]))
X = np.swapaxes(np.array(X), 0, 1)
