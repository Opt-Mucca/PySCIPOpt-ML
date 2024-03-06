import numpy as np
from pyscipopt import Model
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from utils import read_csv_to_dict

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we take the point of view of a health organisation.
This health organisation has a limited amount of supplies for treating water,
and they have the goal to treat as much water as possible such that the treated water
becomes drinkable. Regrettably, they lack resources to explicitly check the potability,
and must rely on some ML model that decides if the water is safe to drink or not.

We have access to open source data
(thanks to: https://github.com/MainakRepositor/Datasets/tree/master and
https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability),
for which we can build predictors. The predictor takes as input a variety of features
about the water, and decides whether or not it is safe to drink.

The goal of this MIP is to treat the samples of water s.t. the largest amount
of water becomes drinkable. There are some constraints on which features can be treated,
due to to available resources.

Let I be the index set of the water samples
Let J be the index set of the features of the water
Let x[i][j] be the variable that takes the value for sample i and feature j after treatment
Let water_sample[i][j] be the value for sample i and feature j before treatment
Let f be the ML predictor that decides if x[i][:] is drinkable or not

The MIP formulation is:

x[i][j] = water_sample[i][j] - removed[i][j] + added[i][j] for all i, j
sum_i removed[i][j] <= remove_budget[j] for all j
sum_i added[i][j] <= added_budget[j] for all j
y[i] = f(x[i][:])

max(sum_i y[i])
"""


def build_and_optimise_water_potability(
    seed=42,
    n_water_samples=50,
    layers_sizes=(10, 10, 10),
    remove_feature_budgets=(2, 20, 200, 1, 20, 20, 1.5, 5, 1),
    add_feature_budgets=(2.2, 22, 220, 1.1, 22, 22, 1.7, 5.5, 1.1),
    framework="sklearn",
    build_only=False,
):
    assert len(layers_sizes) == 3
    assert len(remove_feature_budgets) == 9
    assert len(add_feature_budgets) == 9

    # Path to water potability data
    data_dict = read_csv_to_dict("./tests/data/water.csv")

    # The features of our predictor. All distance based features are variables.
    features = [
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity",
    ]
    n_features = len(features)
    np.random.seed(seed)

    # Generate the actual input data arrays for the ML predictors
    X = []
    y = np.array([int(x) for x in data_dict["Potability"]]).reshape(-1)
    for feature in features:
        X.append(np.array([float(x) for x in data_dict[feature]]))
    X = np.swapaxes(np.array(X), 0, 1)

    # Get the indices of some water that is currently undrinkable
    undrinkable_water_indices = (y == 0).nonzero()[0]
    np.random.shuffle(undrinkable_water_indices)
    undrinkable_water_indices = undrinkable_water_indices[:n_water_samples]

    # Build the MLP classifier
    if framework == "sklearn":
        clf = MLPClassifier(
            random_state=seed,
            hidden_layer_sizes=(layers_sizes[0], layers_sizes[1], layers_sizes[2]),
        ).fit(X, y)
    elif framework == "keras":
        keras.utils.set_random_seed(seed)
        clf = keras.Sequential()
        clf.add(keras.Input(shape=(n_features,)))
        clf.add(keras.layers.Dense(layers_sizes[0], activation="relu"))
        clf.add(keras.layers.Dense(layers_sizes[1], activation="relu"))
        clf.add(keras.layers.Dense(layers_sizes[2], activation="relu"))
        clf.add(keras.layers.Dense(2, activation="linear"))
        clf.add(keras.layers.Activation(keras.activations.softmax))
        clf.compile(optimizer="adam", loss="binary_crossentropy")
        y_keras = np.zeros((len(y), 2))
        for i, class_val in enumerate(y):
            y_keras[i][class_val] = 1
        clf.fit(X, y_keras, batch_size=256, epochs=50)
    else:
        raise ValueError(f"Unknown framework: {framework}")

    # Create the SCIP Model
    scip = Model()

    # Create variables deciding the features of each water sample after treatment
    feature_vars = np.zeros((n_water_samples, n_features), dtype=object)
    features_removed = np.zeros((n_water_samples, n_features), dtype=object)
    features_added = np.zeros((n_water_samples, n_features), dtype=object)
    if framework == "sklearn":
        drinkable_water = np.zeros((n_water_samples, 1), dtype=object)
    else:
        drinkable_water = np.zeros((n_water_samples, 2), dtype=object)

    for i in range(n_water_samples):
        for j in range(drinkable_water.shape[-1]):
            drinkable_water[i][j] = scip.addVar(vtype="B", name=f"drinkable_{i}_{j}")
        for j in range(n_features):
            feature_vars[i][j] = scip.addVar(vtype="C", name=f"feature_val_{i}_{j}")
            features_added[i][j] = scip.addVar(
                vtype="C", lb=0, ub=add_feature_budgets[j], name=f"feature_add_{i}_{j}"
            )
            features_removed[i][j] = scip.addVar(
                vtype="C", lb=0, ub=remove_feature_budgets[j], name=f"feature_rem_{i}_{j}"
            )

    for i in range(n_water_samples):
        for j, feature in enumerate(features):
            scip.addCons(
                data_dict[feature][undrinkable_water_indices[i]]
                - features_removed[i][j]
                + features_added[i][j]
                == feature_vars[i][j],
                name=f"change_{i}_{j}",
            )

    for j, feature in enumerate(features):
        scip.addCons(
            sum(features_added[i][j] for i in range(n_water_samples)) <= add_feature_budgets[j],
            name=f"add_budget_{feature}",
        )
        scip.addCons(
            sum(features_removed[i][j] for i in range(n_water_samples))
            <= remove_feature_budgets[j],
            name=f"remove_budget_{feature}",
        )

    # Add the ML predictor for predicting water quality of the sample
    if framework == "sklearn":
        pred_cons = add_predictor_constr(
            scip,
            clf,
            feature_vars,
            drinkable_water,
            unique_naming_prefix=f"clf_",
        )
    else:
        pred_cons = add_predictor_constr(
            scip,
            clf,
            feature_vars,
            drinkable_water,
            unique_naming_prefix=f"clf_",
            output_type="classification",
        )

    # Set the object to maximise the amount of drinkable water after treatment
    if framework == "sklearn":
        scip.setObjective(-np.sum(drinkable_water) + n_water_samples)
    else:
        scip.setObjective(-np.sum(drinkable_water[:, 1]) + n_water_samples)

    if not build_only:
        # Optimise the SCIP Model
        scip.optimize()

        # We can check the "error" of the MIP embedding via the difference between SKLearn and SCIP output
        if np.max(pred_cons.get_error()) > 10**-3:
            error = np.max(pred_cons.get_error())
            # TODO: There is currently no way to ensure SCIP numerical tolerances dont incorrectly classify

    return scip


def test_water_potability_sklearn():
    scip = build_and_optimise_water_potability(
        seed=42,
        n_water_samples=50,
        layers_sizes=(10, 10, 10),
        remove_feature_budgets=(2, 20, 200, 1, 20, 20, 1.5, 5, 1),
        add_feature_budgets=(2.2, 22, 220, 1.1, 22, 22, 1.7, 5.5, 1.1),
        framework="sklearn",
    )


def test_water_potability_keras():
    scip = build_and_optimise_water_potability(
        seed=42,
        n_water_samples=50,
        layers_sizes=(12, 15, 17),
        remove_feature_budgets=(2, 20, 200, 1, 20, 20, 1.5, 5, 1),
        add_feature_budgets=(2.2, 22, 220, 1.1, 22, 22, 1.7, 5.5, 1.1),
        framework="keras",
    )
