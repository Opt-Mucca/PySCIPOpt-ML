from test_wine_manufacturer import build_and_optimise_wine_manufacturer
from test_adversarial_example import build_and_optimise_adversarial_mnist_torch
from test_auto_manufacturer import build_and_optimise_auto_manufacturer
from test_city_manager import build_and_optimise_city_manager
from test_simple_function_approximation import build_and_optimise_function_approximation_model
from test_tree_planting import build_and_optimise_tree_planting
from test_water_potability import build_and_optimise_water_potability
from test_wine_manufacturer import build_and_optimise_wine_manufacturer

wine_manufacturer_runs = [
    (40, 35, 5, 4.25, "sklearn", "rf", 3, 3, 0.0001),
    (41, 50, 5, 4.15, "sklearn", "rf", 3, 4, 0.0001),
    (42, 45, 6, 4.25, "sklearn", "rf", 4, 3, 0.0001),
    (43, 35, 5, 4.40, "sklearn", "gbdt", 5, 3, 0.0001),
    (44, 40, 6, 4.45, "sklearn", "gbdt", 3, 3, 0.0001),
    (45, 35, 5, 4.25, "sklearn", "gbdt", 3, 5, 0.0001),
    (46, 42, 6, 4.2, "xgboost", "rf", 2, 4, 0.0001),
    (47, 35, 7, 4.25, "xgboost", "rf", 3, 3, 0.0001),
    (48, 45, 5, 4.30, "xgboost", "gbdt", 2, 4, 0.0001),
    (49, 38, 6, 4.15, "xgboost", "gbdt", 3, 3, 0.0001),
    (50, 48, 8, 4.4, "lightgbm", "rf", 2, 3, 0.0001),
    (51, 35, 6, 4.60, "lightgbm", "rf", 3, 3, 0.0001),
    (52, 50, 6, 4.30, "lightgbm", "gbdt", 5, 2, 0.0001),
    (53, 45, 6, 4.20, "lightgbm", "gbdt", 3, 3, 0.0001),
]

summarised_wine_results = []
for i, test_arguments in enumerate(wine_manufacturer_runs):
    if i not in [12, 13]:
        continue
    seed = test_arguments[0]
    n_vineyards = test_arguments[1]
    n_wines_to_produce = test_arguments[2]
    min_wine_quality = test_arguments[3]
    ml_framework = test_arguments[4]
    gbdt_or_rf = test_arguments[5]
    n_estimators = test_arguments[6]
    max_depth = test_arguments[7]
    scip = build_and_optimise_wine_manufacturer(
        seed=seed,
        n_vineyards=n_vineyards,
        n_wines_to_produce=n_wines_to_produce,
        min_wine_quality=min_wine_quality,
        sklearn_xgboost_lightgbm=ml_framework,
        gbdt_or_rf=gbdt_or_rf,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    summarised_wine_results.append(
        (i, scip.getStatus(), scip.getSolvingTime(), scip.getNTotalNodes())
    )

print(summarised_wine_results)
