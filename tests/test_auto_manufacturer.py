import numpy as np
from pyscipopt import Model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from utils import read_csv_to_dict, train_torch_neural_network

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we take the point of view of an auto manufacturer.

We have access to open source data (thanks to: https://www.kaggle.com/datasets/gagandeep16/car-sales),
for which we can build predictors. The predictors takes as input a variety of features for the
given vehicle, and output either the price, the resell value, or the number of units that will sell.
These properties are dependent on each other, so the output of one predictor can also be
input into another predictor.

The goal of this MIP is to design a vehicle that will sell the maximum amount of units possible,
while being sufficiently different from any of the other most popular vehicles.
Additional constraints are added to ensure that the designed vehicle has a high resell value,
and is fuel efficient.

Let x be the vector of static features of the designed vehicle, i.e., those that cannot be changed.
Price, resell value, and amount sold for instance are not entirely controllable, but engine size is.

Let f be the ML predictor that predicts the amount sold of the designed vehicle.
Let g be the ML predictor that predicts the price of the designed vehicle.
Let h be the ML predictor that predicts the resell value of the designed vehicle.

Let V be the index set of best selling vehicles that the designed vehicle must be sufficiently different to.
Let I be the index set of features
Let auto_val[v][i] be the value of feature i for best selling vehicle v
Let e be some minimum level of fuel efficiency
Let c be some minimum ratio of resale value compared to the actual fresh purhcase price

To ensure sufficiently different we use look at the relative difference of each feature.
So for a given vehicle v, and feature i, we know that it has value 5. We also know that feature i must
be in the range [2, 10]. So if our designed vehicle has value 6, then the difference is abs(5-6) / (10-2)

x[i] - auto_val[v][i] == pos_slack[v][i] - neg_slack[v][i] for all V, I
pos_slack[v][i] <= (x[i].UB - x[i].LB) * bin_slack[v][i] for all V, I
neg_slack[v][i] <= (x[i].UB - x[i].LB) * (1 - bin_slack[v][i]) for all V, I
sum_diff[v] = sum_i (pos_slack[v][i] + neg_slack[v][i]) / (x[i].UB - x[i].LB) for all V
sum_diff[v] >= 2 for all V

x[i] >= e where i is the fuel efficiency index

price = g(x)
amount_sold = f(x, price)
resale = h(x, price)

resale >= c * price

max(amount_sold)
"""


def build_and_optimise_auto_manufacturer(
    training_seed=42,
    data_seed=42,
    gbdt_rf_or_mlp="gbdt",
    formulation="sos",
    framework="sklearn",
    max_depth_or_layer_size=6,
    n_estimators_or_layers=6,
    build_only=False,
):
    # Initialise a numpy random state
    data_random_state = np.random.RandomState(data_seed)
    training_random_state = np.random.RandomState(training_seed)

    # Path to car data
    data_dict = read_csv_to_dict("./tests/data/car_sales.csv")

    # Get an array of the features
    features = [
        "Vehicle_type",
        "Engine_size",
        "Horsepower",
        "Wheelbase",
        "Width",
        "Length",
        "Curb_weight",
        "Fuel_capacity",
        "Fuel_efficiency",
        "Power_perf_factor",
    ]
    n_features = len(features)
    n_car_comparisons = 20

    amount_sold_key = "Sales_in_thousands"
    price_key = "Price_in_thousands"
    resale_value_key = "__year_resale_value"

    # Generate the actual input data arrays for the ML predictors
    X = []
    amount_sales = np.array([float(x) for x in data_dict[amount_sold_key]]).reshape(-1, 1)
    price = np.array([float(x) for x in data_dict[price_key]]).reshape(-1, 1)
    resale_price = np.array([float(x) for x in data_dict[resale_value_key]]).reshape(-1, 1)
    for feature in features:
        if feature == "Vehicle_type":
            X.append((np.array(data_dict[feature]) == "Passenger").astype(int))
        else:
            X.append(np.array([float(x) for x in data_dict[feature]]))
    X = np.swapaxes(np.array(X), 0, 1)

    # Create the prediction models
    if gbdt_rf_or_mlp == "gbdt":
        reg_sales = GradientBoostingRegressor(
            n_estimators=n_estimators_or_layers,
            max_depth=max_depth_or_layer_size,
            random_state=training_random_state,
        ).fit(np.concatenate((X, price), axis=1), amount_sales.reshape(-1))
        reg_price = GradientBoostingRegressor(
            n_estimators=n_estimators_or_layers,
            max_depth=max_depth_or_layer_size,
            random_state=training_random_state,
        ).fit(X, price.reshape(-1))
        reg_resale = GradientBoostingRegressor(
            n_estimators=n_estimators_or_layers,
            max_depth=max_depth_or_layer_size,
            random_state=training_random_state,
        ).fit(np.concatenate((X, price), axis=1), resale_price.reshape(-1))
    elif gbdt_rf_or_mlp == "rf":
        reg_sales = RandomForestRegressor(
            n_estimators=n_estimators_or_layers,
            max_depth=max_depth_or_layer_size,
            random_state=training_random_state,
        ).fit(np.concatenate((X, price), axis=1), amount_sales.reshape(-1))
        reg_price = RandomForestRegressor(
            n_estimators=n_estimators_or_layers,
            max_depth=max_depth_or_layer_size,
            random_state=training_random_state,
        ).fit(X, price.reshape(-1))
        reg_resale = RandomForestRegressor(
            n_estimators=n_estimators_or_layers,
            max_depth=max_depth_or_layer_size,
            random_state=training_random_state,
        ).fit(np.concatenate((X, price), axis=1), resale_price.reshape(-1))
    elif gbdt_rf_or_mlp == "mlp":
        if framework == "sklearn":
            hidden_layer_sizes = tuple(
                [max_depth_or_layer_size for i in range(n_estimators_or_layers)]
            )
            reg_sales = MLPRegressor(
                random_state=training_random_state, hidden_layer_sizes=hidden_layer_sizes
            ).fit(np.concatenate((X, price), axis=1), amount_sales.reshape(-1))
            reg_price = MLPRegressor(
                random_state=training_random_state, hidden_layer_sizes=hidden_layer_sizes
            ).fit(X, price.reshape(-1))
            reg_resale = MLPRegressor(
                random_state=training_random_state, hidden_layer_sizes=hidden_layer_sizes
            ).fit(np.concatenate((X, price), axis=1), resale_price.reshape(-1))
        else:
            reg_price = train_torch_neural_network(
                X,
                price,
                n_estimators_or_layers,
                max_depth_or_layer_size,
                training_seed,
                reshape=False,
            )
            reg_sales = train_torch_neural_network(
                np.concatenate((X, price), axis=1),
                amount_sales,
                n_estimators_or_layers,
                max_depth_or_layer_size,
                training_seed,
                reshape=False,
            )
            reg_resale = train_torch_neural_network(
                np.concatenate((X, price), axis=1),
                resale_price,
                n_estimators_or_layers,
                max_depth_or_layer_size,
                training_seed,
                reshape=False,
            )
    else:
        raise ValueError(f"Unknown value: {gbdt_rf_or_mlp}")

    # Create the SCIP Model
    scip = Model()

    # Create variables deciding the features of the manufactured car and the predicted sales information
    feature_vars = np.zeros((1, n_features), dtype=object)
    amount_sold_vars = np.zeros((1, 1), dtype=object)
    price_vars = np.zeros((1, 1), dtype=object)
    resale_vars = np.zeros((1, 1), dtype=object)

    for i in range(n_features):
        if i == 0:
            feature_vars[0][i] = scip.addVar(vtype="B", name=f"feature_{i}")
        else:
            max_val = np.max(X[:, i])
            min_val = np.min(X[:, i])
            lb = max(0, min_val - 0.1 * max_val)
            ub = 1.1 * max_val
            feature_vars[0][i] = scip.addVar(vtype="C", lb=lb, ub=ub, name=f"feature_{i}")

    amount_sold_vars[0][0] = scip.addVar(
        vtype="C", lb=0, ub=(1.1 * np.max(amount_sales)), name="amount_sold"
    )
    price_vars[0][0] = scip.addVar(
        vtype="C",
        lb=(max(0, np.min(price) - 0.1 * np.max(price))),
        ub=(1.1 * np.max(price)),
        name="price",
    )
    resale_vars[0][0] = scip.addVar(
        vtype="C",
        lb=(max(0, np.min(resale_price) - 0.1 * np.max(resale_price))),
        ub=(1.1 * np.max(resale_price)),
        name="resale_value",
    )

    # Create variables used for constraints that ensure the new car is sufficiently different from others
    pos_slack = np.zeros((n_features, n_car_comparisons), dtype=object)
    neg_slack = np.zeros((n_features, n_car_comparisons), dtype=object)
    binary_slack = np.zeros((n_features, n_car_comparisons), dtype=object)
    for i in range(1, n_features):
        for j in range(n_car_comparisons):
            big_m = feature_vars[0][i].getUbOriginal() - feature_vars[0][i].getLbOriginal()
            pos_slack[i][j] = scip.addVar(vtype="C", lb=0, ub=big_m, name=f"pos_slack_{i}_{j}")
            neg_slack[i][j] = scip.addVar(vtype="C", lb=0, ub=big_m, name=f"neg_slack_{i}_{j}")
            binary_slack[i][j] = scip.addVar(vtype="B", name=f"bin_slack_{i}_{j}")

    # Now add constraints to the model. Our constraints are that the new car is sufficiently different from those
    # already on the market, and that the resale value remains relatively high.
    top_selling_ids = np.argsort(price.reshape(-1))[-n_car_comparisons:]
    min_diff = data_random_state.uniform(low=2.8, high=3.3)
    for j, car_idx in enumerate(top_selling_ids):
        sum_slack = 0
        for i in range(1, n_features):
            big_m = feature_vars[0][i].getUbOriginal() - feature_vars[0][i].getLbOriginal()
            scip.addCons(
                pos_slack[i][j] <= big_m * binary_slack[i][j], name=f"bound_pos_slack_{i}_{j}"
            )
            scip.addCons(
                neg_slack[i][j] <= big_m * (1 - binary_slack[i][j]),
                name=f"bound_neg_slack_{i}_{j}",
            )
            feature_val = X[car_idx][i]
            scip.addCons(
                feature_vars[0][i] - feature_val == pos_slack[i][j] - neg_slack[i][j],
                name=f"l1_diff_{i}_{j}",
            )
            sum_slack += (pos_slack[i][j] + neg_slack[i][j]) / big_m
        scip.addCons(sum_slack >= min_diff, name=f"min_diff_{j}")

    # Now add the ML predictor constraints
    pred_cons = [
        add_predictor_constr(
            scip,
            reg_sales,
            np.concatenate((feature_vars, price_vars), axis=1),
            amount_sold_vars,
            epsilon=0.0001,
            formulation=formulation,
            unique_naming_prefix="num_sales_",
        ),
        add_predictor_constr(
            scip,
            reg_price,
            feature_vars,
            price_vars,
            epsilon=0.0001,
            formulation=formulation,
            unique_naming_prefix="price_",
        ),
        add_predictor_constr(
            scip,
            reg_resale,
            np.concatenate((feature_vars, price_vars), axis=1),
            resale_vars,
            epsilon=0.0001,
            formulation=formulation,
            unique_naming_prefix="resale_",
        ),
    ]

    # Add constraints that the resale value of the car must be sufficiently large and that the car is fuel efficient
    min_resale_ratio = data_random_state.uniform(low=0.70, high=0.75)
    min_fuel_efficiency = data_random_state.uniform(low=38, high=43)
    scip.addCons(resale_vars[0][0] >= min_resale_ratio * price_vars[0][0], name="min_resale")
    scip.addCons(feature_vars[0][8] >= min_fuel_efficiency, name="fuel_efficient")

    scip.setObjective(-amount_sold_vars[0][0] + 10000)

    if not build_only:
        # Optimise the SCIP model
        scip.optimize()
        # We can check the "error" of the MIP embedding by determining the difference SKLearn and SCIP output
        for predictor_constraint in pred_cons:
            if np.max(predictor_constraint.get_error()) > 10**-4:
                error = np.max(predictor_constraint.get_error())
                raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -4}")

    return scip


def test_gbdt_auto_manufacturer():
    scip = build_and_optimise_auto_manufacturer(
        training_seed=42,
        data_seed=42,
        gbdt_rf_or_mlp="gbdt",
        max_depth_or_layer_size=5,
        n_estimators_or_layers=5,
        build_only=False,
    )


def test_rf_auto_manufacturer():
    scip = build_and_optimise_auto_manufacturer(
        training_seed=42,
        data_seed=42,
        gbdt_rf_or_mlp="rf",
        max_depth_or_layer_size=5,
        n_estimators_or_layers=5,
        build_only=False,
    )


def test_mlp_auto_manufacturer():
    scip = build_and_optimise_auto_manufacturer(
        training_seed=42,
        data_seed=42,
        gbdt_rf_or_mlp="mlp",
        max_depth_or_layer_size=6,
        n_estimators_or_layers=2,
        build_only=False,
    )


def test_mlp_auto_manufacturer_bigm():
    scip = build_and_optimise_auto_manufacturer(
        training_seed=42,
        data_seed=42,
        gbdt_rf_or_mlp="mlp",
        formulation="bigm",
        max_depth_or_layer_size=6,
        n_estimators_or_layers=2,
        build_only=False,
    )
