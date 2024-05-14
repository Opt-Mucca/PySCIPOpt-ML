import numpy as np
from pyscipopt import Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from utils import read_csv_to_dict, train_torch_neural_network

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we take the point of view of a city manager in Poland.

We have access to open source data
(thanks to: https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland),
for which we have built a predictor. The predictor takes as input a variety of apartment features,
and outputs the price of the apartment, which we consider as a proxy for living quality.

The goal of this MIP is to maximise the living quality of the randomly sampled 50 residents by
building two new schools, clinics, post offices, kindergartens, restaurants,
and pharmacies, as well as a single college. For the sake simplicity we assume the
closest access for all these residents to all these facilities is over 10km away.
To further simplify the MIP we measure distance via the L1 norm at all times.

Let f to be the ML predictor. Let I be the index set of residents and their apartments.
Let J be the index of the building type and nJ be the amount of that building type that will be built.
Let k be the index set of the imaginary 2D grid that will be used for L1 distance.

Let a_loc[i][k] be the location of the apartment i in dimension k

Let build[J][nJ][k] be the variables representing the location of the buildings.
Let x[I][J] be the min distance between the buildings and apartments
Let y[i] be the price of the apartment

The MIP we model below in a city of size 5kmx5km is:

For all building types where two are built ensure a minimum distance between each:
    build[J][0][k] - build[J][1][k] == pos_slack[J][k] - neg_slack[J][k] for all J where nJ == 2 for all k
    pos_slack[J][k] <= 10 * bin_slack[J][k] for all J where nJ == 2 for all k
    neg_slack[J][k] <= 10 * (1 - bin_slack[J][k]) for all J where nJ == 2 for all k
    sum_k pos_slack[J][k] + neg_slack[J][k] >= 1 for all J where nJ == 2

For all apartments track the minimum distance between each building

build[j][nJ][k] - a_loc[i][k] == pos_slack[i][j][nJ][k] - neg_slack[i][j][nJ][k] for all I, J, nJ, k
pos_slack[i][j][nJ][k] <= 10 * bin_slack[i][j][nJ][k] for all I, J, nJ, k
neg_slack[i][j][nJ][k] <= 10 * (1 - bin_slack[i][j][nJ][k]) for all I, J, nJ, k
dist_a_b_k[i][j][nJ][k] = pos_slack[i][j][nJ][k] + neg_slack[i][j][nJ][k] for all I, J, nJ, k
dist_a_b[i][j][nJ] = sum_k dist_a_b_k[i][j][nJ][k] for all I, J, nJ

if nJ == 1
x[i][j] = dist_a_b[i][j][0]

if nJ == 2
dist_a_b[i][j][1] - dist_a_b[i][j][0] <= 10 * bin_slack[i][j] for all I, J
dist_a_b[i][j][0] - dist_a_b[i][j][1] <= 10 * (1 - bin_slack[i][j]) for all I, J
x[i][j] <= dist_a_b[i][j][0] for all I, J
x[i][j] <= dist_a_b[i][j][1] for all I, J
x[i][j] >= dist_a_b[i][j][0] - 10 * (1 - bin_slack[i][j]) for all I, J
x[i][j] >= dist_a_b[i][j][1] - 10 bin_slack[i][j] for all I, J

The embedded ML constraints also need to be added
y[i] = f(x[i][:]) for all I

max(sum_i y[i])
"""


def build_and_optimise_city_manager(
    data_seed=42,
    training_seed=42,
    dt_gbdt_or_mlp="dt",
    formulation="sos",
    framework="sklearn",
    max_depth_or_layer_size=6,
    n_estimators_layers=3,
    n_apartments=25,
    epsilon=0.001,
    build_only=False,
):
    # Path to apartment price data
    data_dict = read_csv_to_dict("./tests/data/apartments.csv")
    grid_length = 5

    # Set the random seed
    data_random_state = np.random.RandomState(data_seed)
    training_random_state = np.random.RandomState(training_seed)

    # The features of our predictor. All distance based features are variables.
    features = [
        "squareMeters",
        "rooms",
        "floorCount",
        "centreDistance",
        "schoolDistance",
        "clinicDistance",
        "postOfficeDistance",
        "kindergartenDistance",
        "restaurantDistance",
        "collegeDistance",
        "pharmacyDistance",
        "hasParkingSpace",
        "hasBalcony",
        "hasSecurity",
    ]
    feature_to_idx = {feature: i for i, feature in enumerate(features)}
    n_features = len(features)

    # Generate the actual input data arrays for the ML predictors
    X = []
    y = np.array([float(x) / 10**5 for x in data_dict["price"]]).reshape(-1, 1)
    for feature in features:
        if "has" in feature:
            X.append(np.array([x == "yes" for x in data_dict[feature]]))
        else:
            X.append(np.array([float(x) for x in data_dict[feature]]))
    X = np.swapaxes(np.array(X), 0, 1)

    # Train the ML predictor
    if dt_gbdt_or_mlp == "dt":
        reg = DecisionTreeRegressor(
            random_state=training_random_state, max_depth=max_depth_or_layer_size
        ).fit(X, y)
    elif dt_gbdt_or_mlp == "gbdt":
        reg = GradientBoostingRegressor(
            random_state=training_random_state,
            max_depth=max_depth_or_layer_size,
            n_estimators=n_estimators_layers,
        ).fit(X, y.reshape(-1))
    elif dt_gbdt_or_mlp == "mlp":
        if framework == "sklearn":
            hidden_layers = tuple([max_depth_or_layer_size for i in range(n_estimators_layers)])
            reg = MLPRegressor(
                random_state=training_random_state, hidden_layer_sizes=hidden_layers
            ).fit(X, y.reshape(-1))
        else:
            reg = train_torch_neural_network(
                X, y, n_estimators_layers, max_depth_or_layer_size, training_seed, reshape=False
            )
    else:
        raise ValueError(f"Unknown value: {dt_gbdt_or_mlp}")

    # Create the SCIP Model
    scip = Model()

    # Create variables for the n_apartments many imaginary apartments.The apartments lie on an imaginary square that is
    # defined by the four corners [(0, 0), (0, grid_length), (grid_length, grid_length), (grid_length, 0)]
    feature_vars = np.zeros((n_apartments, n_features), dtype=object)
    price_vars = np.zeros((n_apartments, 1), dtype=object)
    apartment_locations = data_random_state.uniform(0, grid_length, size=(n_apartments, 2))
    rooms = np.random.randint(
        np.min(X[feature_to_idx["rooms"]]),
        np.max(X[feature_to_idx["rooms"]]) + 1,
        size=n_apartments,
    )
    floors = np.random.randint(
        np.min(X[feature_to_idx["floorCount"]]),
        np.max(X[feature_to_idx["floorCount"]]) + 1,
        size=n_apartments,
    )
    square_meters = np.random.uniform(
        np.min(X[feature_to_idx["squareMeters"]]),
        np.max(X[feature_to_idx["squareMeters"]]),
        size=n_apartments,
    )

    # Create variables for where the various buildings will be placed
    school_vars = np.zeros((2, 2), dtype=object)
    clinic_vars = np.zeros((2, 2), dtype=object)
    post_office_vars = np.zeros((2, 2), dtype=object)
    kindergarten_vars = np.zeros((2, 2), dtype=object)
    restaurant_vars = np.zeros((2, 2), dtype=object)
    college_vars = np.zeros((1, 2), dtype=object)
    pharmacy_vars = np.zeros((2, 2), dtype=object)
    building_vars = [
        ("school", "schoolDistance", school_vars),
        ("clinic", "clinicDistance", clinic_vars),
        ("post_office", "postOfficeDistance", post_office_vars),
        ("kindergarten", "kindergartenDistance", kindergarten_vars),
        ("restaurant", "restaurantDistance", restaurant_vars),
        ("college", "collegeDistance", college_vars),
        ("pharmacy", "pharmacyDistance", pharmacy_vars),
    ]

    # Now fill in the actual variables
    for i in range(n_apartments):
        price_vars[i][0] = scip.addVar(vtype="C", lb=-25, ub=100, name=f"price_{i}")
        for j, feature in enumerate(features):
            feature_vars[i][j] = scip.addVar(vtype="C", lb=0, name=f"feature_{i}_{j}")
            # Randomly generate characteristics
            if "has" in feature:
                scip.fixVar(feature_vars[i][j], int(np.random.randint(0, 2)))
            elif feature == "rooms":
                scip.fixVar(feature_vars[i][j], int(rooms[i]))
            elif feature == "floorCount":
                scip.fixVar(feature_vars[i][j], int(floors[i]))
            elif feature == "squareMeters":
                scip.fixVar(feature_vars[i][j], int(square_meters[i]))
            elif feature == "centreDistance":
                scip.fixVar(
                    feature_vars[i][j],
                    np.sum(
                        np.abs(
                            apartment_locations[i] - np.array([grid_length / 2, grid_length / 2])
                        )
                    ),
                )
            else:
                scip.chgVarUb(feature_vars[i][j], 2 * grid_length)

    for building_type, _, building_var in building_vars:
        for i in range(building_var.shape[0]):
            building_var[i][0] = scip.addVar(
                vtype="C", lb=0, ub=grid_length, name=f"{building_type}_{i}_0"
            )
            building_var[i][1] = scip.addVar(
                vtype="C", lb=0, ub=grid_length, name=f"{building_type}_{i}_1"
            )

    # Add constraints that link the distance of the built structures to each apartment
    for building_type, feature, building_var in building_vars:
        feature_idx = feature_to_idx[feature]
        n_buildings = building_var.shape[0]
        big_m = grid_length * 2

        # If there are two buildings ensure that there is a minimum distance between them
        if n_buildings == 2:
            pos_dists = [
                scip.addVar(vtype="C", lb=0, ub=big_m, name=f"pos__{building_type}_{i}_{k}")
                for k in range(2)
            ]
            neg_dists = [
                scip.addVar(vtype="C", lb=0, ub=big_m, name=f"neg_{building_type}_{i}_{k}")
                for k in range(2)
            ]
            bin_dists = [
                scip.addVar(vtype="B", name=f"bin_{building_type}_{i}_{k}") for k in range(2)
            ]
            for k in range(2):
                scip.addCons(
                    building_var[0][k] - building_var[1][k] == pos_dists[k] - neg_dists[k],
                    name=f"dist_build_{building_type}_{i}_{k}",
                )
                scip.addCons(
                    pos_dists[k] <= big_m * bin_dists[k],
                    name=f"big_m_pos_{building_type}_{i}_{k}",
                )
                scip.addCons(
                    neg_dists[k] <= big_m * (1 - bin_dists[k]),
                    name=f"big_m_neg_{building_type}_{i}_{k}",
                )
            scip.addCons(
                sum(pos_dists[k] + neg_dists[k] for k in range(2)) >= 0.8,
                name=f"min_dist_build_{building_type}",
            )

        # Add constraints that measure the distance between each apartment and the placed building
        for i in range(n_apartments):
            if building_var.shape[0] > 2:
                raise AssertionError(
                    "Didn't program formulations for more than 2 buildings of the same type"
                )

            dist_builds = [
                scip.addVar(
                    vtype="C", lb=0, ub=2 * grid_length, name=f"sum_dist_{building_type}_{i}_{j}"
                )
                for j in range(n_buildings)
            ]
            dist_dims = [
                [
                    scip.addVar(
                        vtype="C", lb=0, ub=big_m, name=f"dist_{building_type}_{i}_{j}_{k}"
                    )
                    for k in range(2)
                ]
                for j in range(n_buildings)
            ]
            pos_dists = [
                [
                    scip.addVar(
                        vtype="C", lb=0, ub=big_m, name=f"pos__{building_type}_{i}_{j}_{k}"
                    )
                    for k in range(2)
                ]
                for j in range(n_buildings)
            ]
            neg_dists = [
                [
                    scip.addVar(vtype="C", lb=0, ub=big_m, name=f"neg_{building_type}_{i}_{j}_{k}")
                    for k in range(2)
                ]
                for j in range(n_buildings)
            ]
            bin_dists = [
                [scip.addVar(vtype="B", name=f"bin_{building_type}_{i}_{j}_{k}") for k in range(2)]
                for j in range(n_buildings)
            ]
            for j in range(n_buildings):
                for k in range(2):
                    scip.addCons(
                        building_var[j][k] - apartment_locations[i][k]
                        == pos_dists[j][k] - neg_dists[j][k],
                        name=f"dist_apartment_{building_type}_{i}_{j}_{k}",
                    )
                    scip.addCons(
                        pos_dists[j][k] <= big_m * bin_dists[j][k],
                        name=f"big_m_pos_{building_type}_{i}_{j}_{k}",
                    )
                    scip.addCons(
                        neg_dists[j][k] <= big_m * (1 - bin_dists[j][k]),
                        name=f"big_m_neg_{building_type}_{i}_{j}_{k}",
                    )
                    scip.addCons(
                        dist_dims[j][k] == pos_dists[j][k] + neg_dists[j][k],
                        name=f"sum_dist_{building_type}_{i}_{j}_{k}",
                    )
                scip.addCons(
                    dist_builds[j] == dist_dims[j][0] + dist_dims[j][1],
                    name=f"sum_dist_dims_{i}_{building_type}",
                )
            if n_buildings == 1:
                scip.addCons(
                    feature_vars[i][feature_idx] == dist_builds[0],
                    name=f"feature_dist_{i}_{feature_idx}",
                )
            else:
                aux_max_var_bin = scip.addVar(vtype="B", name=f"aux_max_{i}_{building_type}")
                scip.addCons(
                    dist_builds[1] - dist_builds[0] <= big_m * aux_max_var_bin,
                    name=f"big_m_{i}_{building_type}_0",
                )
                scip.addCons(
                    dist_builds[0] - dist_builds[1] <= big_m * (1 - aux_max_var_bin),
                    name=f"big_m_{i}_{building_type}_1",
                )
                scip.addCons(
                    feature_vars[i][feature_idx] <= dist_builds[0],
                    name=f"dist_{i}_{building_type}_0",
                )
                scip.addCons(
                    feature_vars[i][feature_idx] <= dist_builds[1],
                    name=f"dist_{i}_{building_type}_1",
                )
                scip.addCons(
                    feature_vars[i][feature_idx] >= dist_builds[0] - big_m * (1 - aux_max_var_bin),
                    name=f"dist_{i}_{building_type}_2",
                )
                scip.addCons(
                    feature_vars[i][feature_idx] >= dist_builds[1] - big_m * aux_max_var_bin,
                    name=f"dist_{i}_{building_type}_3",
                )

    # Embed the ML predictors for each apartment. A small epsilon ensures we match sklearn
    pred_cons = add_predictor_constr(
        scip,
        reg,
        feature_vars,
        price_vars,
        epsilon=epsilon,
        formulation=formulation,
        unique_naming_prefix=f"predictor_",
    )

    # Add the objective to the MIP
    scip.setObjective(-np.sum(price_vars) + (20 * n_apartments))

    if not build_only:
        # Optimise the SCIP model
        scip.optimize()
        # We can check the "error" of the MIP embedding by determining the difference SKLearn and SCIP output
        if np.max(pred_cons.get_error()) > 10**-4:
            error = np.max(pred_cons.get_error())
            raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -4}")

    return scip


def test_dt_city_manager():
    build_and_optimise_city_manager(
        data_seed=42,
        training_seed=42,
        dt_gbdt_or_mlp="dt",
        max_depth_or_layer_size=6,
        n_estimators_layers=3,
        n_apartments=25,
        epsilon=0.001,
        build_only=False,
    )


def test_gbdt_city_manager():
    build_and_optimise_city_manager(
        data_seed=50,
        training_seed=80,
        dt_gbdt_or_mlp="gbdt",
        max_depth_or_layer_size=4,
        n_estimators_layers=3,
        n_apartments=25,
        epsilon=0.001,
        build_only=False,
    )


def test_mlp_city_manager():
    build_and_optimise_city_manager(
        data_seed=50,
        training_seed=80,
        dt_gbdt_or_mlp="mlp",
        formulation="sos",
        max_depth_or_layer_size=3,
        n_estimators_layers=2,
        n_apartments=3,
        epsilon=0.001,
        build_only=False,
    )


def test_mlp_bigm_city_manager():
    build_and_optimise_city_manager(
        data_seed=50,
        training_seed=80,
        dt_gbdt_or_mlp="mlp",
        formulation="bigm",
        max_depth_or_layer_size=3,
        n_estimators_layers=2,
        n_apartments=3,
        epsilon=0.001,
        build_only=False,
    )
