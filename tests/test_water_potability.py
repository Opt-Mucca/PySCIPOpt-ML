import numpy as np
from pyscipopt import Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from utils import read_csv_to_dict, train_torch_neural_network

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
    data_seed=42,
    training_seed=42,
    predictor_type="mlp",
    formulation="sos",
    n_water_samples=50,
    layer_size=16,
    max_depth=5,
    n_estimators_layers=3,
    framework="sklearn",
    build_only=False,
):
    assert predictor_type in ("mlp", "gbdt")
    # Random seed initialisation
    data_random_state = np.random.RandomState(data_seed)
    training_random_state = np.random.RandomState(training_seed)
    remove_feature_budgets = (
        data_random_state.uniform(1.8, 2.5),
        data_random_state.uniform(20, 25),
        data_random_state.uniform(180, 250),
        data_random_state.uniform(0.9, 1.1),
        data_random_state.uniform(19, 21),
        data_random_state.uniform(19, 21),
        data_random_state.uniform(1.2, 1.8),
        data_random_state.uniform(4.5, 6),
        data_random_state.uniform(0.9, 1.1),
    )
    add_feature_budgets = (
        data_random_state.uniform(1.8, 2.5),
        data_random_state.uniform(20, 25),
        data_random_state.uniform(180, 250),
        data_random_state.uniform(0.9, 1.1),
        data_random_state.uniform(19, 21),
        data_random_state.uniform(19, 21),
        data_random_state.uniform(1.2, 1.8),
        data_random_state.uniform(4.5, 6),
        data_random_state.uniform(0.9, 1.1),
    )

    # Path to water potability data
    data_dict = read_csv_to_dict("./tests/data/water_quality.csv")

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
    if predictor_type == "gbdt":
        clf = GradientBoostingClassifier(
            random_state=training_random_state,
            max_depth=max_depth,
            n_estimators=n_estimators_layers,
        ).fit(X, y)
    else:
        if framework == "sklearn":
            hidden_layers = tuple([layer_size for i in range(n_estimators_layers)])
            clf = MLPClassifier(
                random_state=training_random_state,
                hidden_layer_sizes=hidden_layers,
            ).fit(X, y)
        elif framework == "keras":
            keras.utils.set_random_seed(training_seed)
            clf = keras.Sequential()
            clf.add(keras.Input(shape=(n_features,)))
            for i in range(n_estimators_layers):
                clf.add(keras.layers.Dense(layer_size, activation="relu"))
            clf.add(keras.layers.Dense(2, activation="linear"))
            clf.add(keras.layers.Activation(keras.activations.softmax))
            clf.compile(optimizer="adam", loss="binary_crossentropy")
            y_keras = np.zeros((len(y), 2))
            for i, class_val in enumerate(y):
                y_keras[i][class_val] = 1
            clf.fit(X, y_keras, batch_size=256, epochs=50)
        elif framework == "torch":
            clf = train_torch_neural_network(
                X,
                y,
                n_estimators_layers,
                layer_size,
                training_seed,
                reshape=True,
                binary_classifier=True,
            )
        else:
            raise ValueError(f"Unknown framework: {framework}")

    # Create the SCIP Model
    scip = Model()

    # Create variables deciding the features of each water sample after treatment
    feature_vars = np.zeros((n_water_samples, n_features), dtype=object)
    features_removed = np.zeros((n_water_samples, n_features), dtype=object)
    features_added = np.zeros((n_water_samples, n_features), dtype=object)
    if framework in ["sklearn", "torch"] or predictor_type == "gbdt":
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
            lb = X[undrinkable_water_indices[i]][j] - remove_feature_budgets[j]
            ub = X[undrinkable_water_indices[i]][j] + add_feature_budgets[j]
            if lb >= 0:
                scip.chgVarLb(feature_vars[i][j], lb)
            scip.chgVarUb(feature_vars[i][j], ub)
            scip.addCons(
                X[undrinkable_water_indices[i]][j] - features_removed[i][j] + features_added[i][j]
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
    if framework == "sklearn" or predictor_type == "gbdt":
        pred_cons = add_predictor_constr(
            scip,
            clf,
            feature_vars,
            drinkable_water,
            unique_naming_prefix=f"clf_",
            epsilon=0.0001,
            formulation=formulation,
        )
    else:
        pred_cons = add_predictor_constr(
            scip,
            clf,
            feature_vars,
            drinkable_water,
            unique_naming_prefix=f"clf_",
            output_type="classification",
            formulation=formulation,
        )

    # Set the object to maximise the amount of drinkable water after treatment
    if framework in ["sklearn", "torch"] or predictor_type == "gbdt":
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


def test_water_potability_sklearn_mlp():
    scip = build_and_optimise_water_potability(
        data_seed=42,
        training_seed=42,
        predictor_type="mlp",
        n_water_samples=50,
        layer_size=16,
        max_depth=5,
        n_estimators_layers=2,
        framework="sklearn",
        build_only=False,
    )


def test_water_potability_sklearn_mlp_bigm():
    scip = build_and_optimise_water_potability(
        data_seed=42,
        training_seed=42,
        predictor_type="mlp",
        formulation="bigm",
        n_water_samples=50,
        layer_size=16,
        max_depth=5,
        n_estimators_layers=2,
        framework="sklearn",
        build_only=False,
    )


def test_water_potability_keras():
    scip = build_and_optimise_water_potability(
        data_seed=20,
        training_seed=20,
        predictor_type="mlp",
        n_water_samples=50,
        layer_size=16,
        max_depth=5,
        n_estimators_layers=2,
        framework="keras",
        build_only=False,
    )


def test_water_potability_keras_bigm():
    scip = build_and_optimise_water_potability(
        data_seed=20,
        training_seed=20,
        predictor_type="mlp",
        formulation="bigm",
        n_water_samples=50,
        layer_size=16,
        max_depth=5,
        n_estimators_layers=2,
        framework="keras",
        build_only=False,
    )


def test_water_potability_gbdt():
    scip = build_and_optimise_water_potability(
        data_seed=18,
        training_seed=18,
        predictor_type="gbdt",
        n_water_samples=50,
        layer_size=16,
        max_depth=4,
        n_estimators_layers=4,
        framework="keras",
        build_only=False,
    )
