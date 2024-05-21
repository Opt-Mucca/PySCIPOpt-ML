import numpy as np
from lightgbm import LGBMRegressor
from pyscipopt import Model, quicksum
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from utils import read_csv_to_dict, train_torch_neural_network
from xgboost import XGBRegressor, XGBRFRegressor

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we take the point of view of a wine manufacturer.

We have access to open source data (thanks to:
@article{cortez2009modeling,
  title={Modeling wine preferences by data mining from physicochemical properties},
  author={Cortez, Paulo and Cerdeira, Ant{\'o}nio and Almeida, Fernando and Matos, Telmo and Reis, Jos{\'e}},
  journal={Decision support systems},
  volume={47},
  number={4},
  pages={547--553},
  year={2009},
  publisher={Elsevier}
}
Data accessed here: https://www.kaggle.com/datasets/piyushgoyal443/red-wine-dataset)
for which we have built a predictor. The predictor takes as input a variety of wine features,
and determines the quality of the wine. This can either be modelled as a regression model,
with the quality being in the range [0,1], or some quality threshold can be set,
and the task made into a classification task.

The goal of this MIP is to create a diverse bouquet of wines with the highest average
quality. To do this, we make an assumption that the features of the grapes and resulting wine
are identical. In this MIP we can purchase grapes from a set of suppliers, all of whom
have grapes that by themselves would result in low quality wine. We can blend the grapes, however,
such that the predicted quality of the wine increases. Each supplier has a limit on the amount of
grapes that can be sold, and an associated cost per unit. Additionally,
we create the same amount of wine for each wine of the bouquet.

Let I be the index set of the wine bouquet that will be created
Let J be the index set of features for each grape / wine.
Let K be the index set of vineyards where the grapes can be purchased
Let x[i][j] be the feature j of wine i
Let y[i] be the quality of the wine i
Let m[i][k] be the amount of grapes from vineyard k used in wine i
let f be the ML predictor that determines the quality of a wine
Let feature_val[j][k] be the value of feature j from vineyard k
Let costs[k] be the cost of one unit of grapes from vineyard k
Let available[k] be the amount of grapes avalable from vineyard k
Let budget be the total budget available

The MIP model is:

x[i][j] = sum_k m[i][k] * feature_val[j][k] for all i, i
sum_k m[i][k] = 1 for all i
sum_i m[i][k] <= available[k]
y[i] = f(x[i][:]) for all i
sum_{i,k} m[i][k] * costs[k] <= budget

max(sum_i y[i])
"""


def build_and_optimise_wine_manufacturer(
    data_seed=42,
    training_seed=42,
    n_vineyards=35,
    n_wines_to_produce=5,
    framework="sklearn",
    gbdt_rf_or_mlp="rf",
    formulation="sos",
    n_estimators_layers=3,
    max_depth=3,
    layer_size=16,
    epsilon=0.0001,
    build_only=False,
):
    assert framework in ("sklearn", "xgboost", "lightgbm", "torch")
    assert gbdt_rf_or_mlp in ("gbdt", "rf", "mlp")
    data_random_state = np.random.RandomState(data_seed)
    training_random_state = np.random.RandomState(training_seed)

    # Path to red wine data
    data_dict = read_csv_to_dict("./tests/data/red_wine_quality.csv")

    features = [
        "fixed.acidity",
        "volatile.acidity",
        "citric.acid",
        "residual.sugar",
        "chlorides",
        "free.sulfur.dioxide",
        "total.sulfur.dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    n_features = len(features)
    budget = 1.7 * n_wines_to_produce

    # Generate the actual input data arrays for the ML predictors
    X = []
    quality = np.array([float(x) for x in data_dict["quality"]]).reshape(
        -1,
    )
    for feature in features:
        X.append(np.array([float(x) for x in data_dict[feature]]))
    X = np.swapaxes(np.array(X), 0, 1)

    # Train the ML predictor
    if gbdt_rf_or_mlp == "mlp":
        if framework == "sklearn":
            hidden_layers = tuple([layer_size for i in range(n_estimators_layers)])
            reg = MLPRegressor(
                random_state=training_random_state, hidden_layer_sizes=hidden_layers
            ).fit(X, quality)
        else:
            reg = train_torch_neural_network(
                X, quality, n_estimators_layers, layer_size, training_seed, reshape=True
            )
    elif framework == "sklearn":
        if gbdt_rf_or_mlp == "rf":
            reg = RandomForestRegressor(
                random_state=training_random_state,
                n_estimators=n_estimators_layers,
                max_depth=max_depth,
            ).fit(X, quality)
        else:
            reg = GradientBoostingRegressor(
                random_state=training_random_state,
                n_estimators=n_estimators_layers,
                max_depth=max_depth,
            ).fit(X, quality)
    elif framework == "xgboost":
        if gbdt_rf_or_mlp == "gbdt":
            reg = XGBRegressor(
                random_state=training_random_state,
                n_estimators=n_estimators_layers,
                max_depth=max_depth,
            ).fit(X, quality)
        else:
            reg = XGBRFRegressor(
                random_state=training_random_state,
                n_estimators=n_estimators_layers,
                max_depth=max_depth,
            ).fit(X, quality)
    else:
        if gbdt_rf_or_mlp == "gbdt":
            reg = LGBMRegressor(
                random_state=training_random_state,
                n_estimators=n_estimators_layers,
                max_depth=max_depth,
            ).fit(X, quality)
        else:
            reg = LGBMRegressor(
                random_state=training_random_state,
                n_estimators=n_estimators_layers,
                max_depth=max_depth,
                boosting_type="rf",
                bagging_freq=1,
                bagging_fraction=0.5,
            ).fit(X, quality)

    # Create artificial data from some vineyards
    vineyard_order = np.arange(X.shape[0])
    data_random_state.shuffle(vineyard_order)
    vineyard_litre_limits = data_random_state.uniform(0.25, 0.35, n_vineyards)
    vineyard_costs = data_random_state.uniform(1, 2, n_vineyards)
    vineyard_features = []
    low_quality_vineyards_i = 0
    for i in vineyard_order:
        if low_quality_vineyards_i >= n_vineyards:
            break
        if quality[i] <= 5:
            low_quality_vineyards_i += 1
            vineyard_features.append(X[i])
    vineyard_features = np.array(vineyard_features)

    # Create the SCIP Model
    scip = Model()

    # Create variables deciding the features of each wine
    feature_vars = np.zeros((n_wines_to_produce, n_features), dtype=object)
    quality_vars = np.zeros((n_wines_to_produce, 1), dtype=object)
    wine_mixture_vars = np.zeros((n_wines_to_produce, n_vineyards), dtype=object)

    for i in range(n_wines_to_produce):
        quality_vars[i][0] = scip.addVar(vtype="C", lb=0, ub=10, name=f"quality_{i}")
        for j in range(n_features):
            max_val = np.max(X[:, j])
            min_val = np.min(X[:, j])
            lb = max(0, min_val - 0.1 * max_val)
            ub = 1.1 * max_val
            feature_vars[i][j] = scip.addVar(vtype="C", lb=lb, ub=ub, name=f"feature_{i}_{j}")
        for k in range(n_vineyards):
            wine_mixture_vars[i][k] = scip.addVar(
                vtype="C", lb=0, ub=vineyard_litre_limits[k], name=f"mixture_{i}_{k}"
            )

    # Now create constraints on the wine blending
    for i in range(n_wines_to_produce):
        for j in range(n_features):
            scip.addCons(
                feature_vars[i][j]
                == quicksum(
                    wine_mixture_vars[i][k] * vineyard_features[k][j] for k in range(n_vineyards)
                ),
                name=f"mixture_cons_{i}_{j}",
            )
    for i in range(n_wines_to_produce):
        scip.addCons(
            quicksum(wine_mixture_vars[i][k] for k in range(n_vineyards)) == 1,
            name=f"wine_mix_{i}",
        )
    for k in range(n_vineyards):
        scip.addCons(
            quicksum(wine_mixture_vars[i][k] for i in range(n_wines_to_produce))
            <= vineyard_litre_limits[k],
            name=f"vineyard_limit_{k}",
        )

    # Add the budget constraint
    scip.addCons(
        quicksum(
            quicksum(wine_mixture_vars[i][k] * vineyard_costs[k] for k in range(n_vineyards))
            for i in range(n_wines_to_produce)
        )
        <= budget,
        name=f"budget_cons",
    )

    # Add the ML constraint. Add in a single batch!
    pred_cons = add_predictor_constr(
        scip,
        reg,
        feature_vars,
        quality_vars,
        unique_naming_prefix="predictor_",
        epsilon=epsilon,
        formulation=formulation,
    )

    # Add a constraint ensuring minimum wine quality on those produced
    min_wine_quality = data_random_state.uniform(4.2, 4.5)
    for i in range(n_wines_to_produce):
        scip.addCons(quality_vars[i][0] >= min_wine_quality, name=f"min_quality_{i}")

    # Set the SCIP objective
    scip.setObjective(
        quicksum(-quality_vars[i][0] for i in range(n_wines_to_produce)) / n_wines_to_produce + 10
    )

    if not build_only:
        # Optimise the SCIP model
        scip.optimize()

        # We can check the "error" of the MIP embedding via the difference between SKLearn and SCIP output
        if np.max(pred_cons.get_error()) > 10**-3:
            error = np.max(pred_cons.get_error())
            raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -3}")

    return scip


def test_wine_manufacturer_sk_rf():
    scip = build_and_optimise_wine_manufacturer(
        data_seed=42,
        training_seed=42,
        n_vineyards=35,
        n_wines_to_produce=5,
        framework="sklearn",
        gbdt_rf_or_mlp="rf",
        max_depth=3,
        n_estimators_layers=3,
    )


def test_wine_manufacturer_sk_gbdt():
    scip = build_and_optimise_wine_manufacturer(
        data_seed=42,
        training_seed=42,
        n_vineyards=35,
        n_wines_to_produce=5,
        framework="sklearn",
        gbdt_rf_or_mlp="gbdt",
        max_depth=3,
        n_estimators_layers=3,
    )


def test_wine_manufacturer_xgb_rf():
    scip = build_and_optimise_wine_manufacturer(
        data_seed=21,
        training_seed=21,
        n_vineyards=35,
        n_wines_to_produce=5,
        framework="xgboost",
        gbdt_rf_or_mlp="rf",
        max_depth=3,
        n_estimators_layers=3,
    )


def test_wine_manufacturer_xgb_gbdt():
    scip = build_and_optimise_wine_manufacturer(
        data_seed=21,
        training_seed=21,
        n_vineyards=35,
        n_wines_to_produce=5,
        framework="xgboost",
        gbdt_rf_or_mlp="gbdt",
        max_depth=3,
        n_estimators_layers=3,
    )


def test_wine_manufacturer_lgb_rf():
    scip = build_and_optimise_wine_manufacturer(
        data_seed=18,
        training_seed=18,
        n_vineyards=35,
        n_wines_to_produce=5,
        framework="lightgbm",
        gbdt_rf_or_mlp="rf",
        max_depth=3,
        n_estimators_layers=3,
    )


def test_wine_manufacturer_lgb_gbdt():
    scip = build_and_optimise_wine_manufacturer(
        data_seed=18,
        training_seed=18,
        n_vineyards=35,
        n_wines_to_produce=5,
        framework="lightgbm",
        gbdt_rf_or_mlp="gbdt",
        max_depth=3,
        n_estimators_layers=3,
    )


def test_wine_manufacturer_sklearn_mlp():
    scip = build_and_optimise_wine_manufacturer(
        data_seed=18,
        training_seed=18,
        n_vineyards=35,
        n_wines_to_produce=5,
        framework="sklearn",
        gbdt_rf_or_mlp="mlp",
        max_depth=3,
        n_estimators_layers=3,
        layer_size=7,
    )


def test_wine_manufacturer_sklearn_mlp_bigm():
    scip = build_and_optimise_wine_manufacturer(
        data_seed=18,
        training_seed=18,
        n_vineyards=35,
        n_wines_to_produce=5,
        framework="sklearn",
        gbdt_rf_or_mlp="mlp",
        formulation="bigm",
        max_depth=3,
        n_estimators_layers=3,
        layer_size=7,
    )
