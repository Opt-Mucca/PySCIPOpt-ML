from test_adversarial_example import build_and_optimise_adversarial_mnist_torch
from test_auto_manufacturer import build_and_optimise_auto_manufacturer
from test_city_manager import build_and_optimise_city_manager
from test_simple_function_approximation import (
    build_and_optimise_function_approximation_model,
)
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
        build_only=True,
    )
    scip.writeProblem(f"tests/data/instances/wine_{i}.mps")
    summarised_wine_results.append(
        (i, scip.getStatus(), scip.getSolvingTime(), scip.getNTotalNodes())
    )
print(summarised_wine_results)

water_potability_runs = [
    (
        40,
        80,
        (20, 16, 20),
        (3, 40, 200, 10, 120, 20, 1.2, 5, 2),
        (8.4, 100, 300, 50, 186, 220, 3, 8.2, 6.4),
    ),
    (
        41,
        82,
        (10, 20, 18),
        (7, 50, 340, 2, 30, 125, 11.5, 5, 11.9),
        (8.2, 50, 320, 2.3, 32, 126, 11.9, 6, 11.3),
    ),
    (
        42,
        70,
        (20, 24, 7),
        (2, 15, 200, 1.2, 20, 20, 1.2, 5, 0.8),
        (2.4, 22, 200, 1.1, 26, 22, 1.7, 5.2, 1.1),
    ),
    (
        43,
        60,
        (24, 20, 30),
        (3, 20, 240, 1, 20, 25, 1.5, 4.6, 1),
        (2.4, 20, 220, 1.3, 22, 26, 1.9, 5.5, 1.3),
    ),
    (
        45,
        40,
        (20, 32, 20),
        (3, 20, 240, 1, 20, 25, 1.5, 4.6, 1),
        (2.4, 20, 220, 1.3, 22, 26, 1.9, 5.5, 1.3),
    ),
    (
        46,
        60,
        (24, 9, 24),
        (2, 15, 200, 1.2, 20, 20, 1.2, 5, 0.8),
        (2.2, 22, 200, 1.1, 26, 22, 1.7, 5.2, 1.1),
    ),
    (
        47,
        55,
        (18, 18, 28),
        (3, 20, 240, 1, 20, 25, 1.5, 4.6, 1),
        (2.4, 20, 220, 1.3, 22, 26, 1.9, 5.5, 1.3),
    ),
]

summarised_water_results = []
for i, test_arguments in enumerate(water_potability_runs):
    seed = test_arguments[0]
    n_water_samples = test_arguments[1]
    layer_sizes = test_arguments[2]
    remove_feature_budgets = test_arguments[3]
    add_feature_budgets = test_arguments[4]

    scip = build_and_optimise_water_potability(
        seed=seed,
        n_water_samples=n_water_samples,
        layers_sizes=layer_sizes,
        remove_feature_budgets=remove_feature_budgets,
        add_feature_budgets=add_feature_budgets,
        build_only=True,
    )
    scip.writeProblem(f"tests/data/instances/water_{i}.mps")
    summarised_water_results.append(
        (i, scip.getStatus(), scip.getSolvingTime(), scip.getNTotalNodes())
    )

print(summarised_water_results)


tree_planting_runs = [
    (42, "linear", 5, 10, [2, 2, 2, 2], 10, [25, 30, 40, 50], 3350),
    (43, "linear", 5, 11, [2.5, 2.5, 2.5, 2.5], 11, [25, 30, 40, 50], 3400),
    (44, "linear", 5, 12, [2, 2, 2, 2], 12, [25, 30, 40, 50], 3900),
    (45, "linear", 5, 13, [2.3, 2.3, 2.3, 2.3], 10, [25, 30, 40, 52], 4300),
    (46, "decision_tree", 6, 14, [5, 3.8, 5.3, 5.6], 5, [25, 30, 40, 50], 5550),
]

summarised_tree_results = []
for i, test_arguments in enumerate(tree_planting_runs):
    seed = test_arguments[0]
    predictor_type = test_arguments[1]
    max_depth = test_arguments[2]
    n_grid_size = test_arguments[3]
    min_trees = test_arguments[4]
    max_sterilise = test_arguments[5]
    costs = test_arguments[6]
    max_budget = test_arguments[7]
    scip = build_and_optimise_tree_planting(
        seed=seed,
        predictor_type=predictor_type,
        max_depth=max_depth,
        n_grid_size=n_grid_size,
        min_trees=min_trees,
        max_sterilise=max_sterilise,
        costs=costs,
        max_budget=max_budget,
        build_only=True,
    )
    scip.writeProblem(f"tests/data/instances/tree_{i}.mps")
    summarised_tree_results.append(
        (i, scip.getStatus(), scip.getSolvingTime(), scip.getNTotalNodes())
    )

print(summarised_tree_results)

function_approximation_runs = [
    (42, 5, 1000, "sklearn", (20, 55, 15)),
    (43, 6, 1000, "sklearn", (30, 40, 25)),
    (44, 7, 1000, "sklearn", (15, 50, 10)),
    (46, 9, 1000, "sklearn", (20, 15, 15)),
    (47, 8, 1000, "torch", (35, 55, 35)),
]

summarised_function_approximation_results = []
for i, test_arguments in enumerate(function_approximation_runs):
    seed = test_arguments[0]
    n_inputs = test_arguments[1]
    n_samples = test_arguments[2]
    sklearn_or_torch = test_arguments[3]
    layer_sizes = test_arguments[4]
    scip = build_and_optimise_function_approximation_model(
        seed=seed,
        n_inputs=n_inputs,
        n_samples=n_samples,
        sklearn_or_torch=sklearn_or_torch,
        layers_sizes=layer_sizes,
        build_only=True,
    )
    scip.writeProblem(f"tests/data/instances/functionapprox_{i}.mps")
    summarised_function_approximation_results.append(
        (i, scip.getStatus(), scip.getSolvingTime(), scip.getNTotalNodes())
    )

print(summarised_function_approximation_results)


city_manager_runs = [
    (42, 5, 60, 5, 0.0001),
    (43, 6, 60, 4, 0.0001),
    (44, 7, 50, 5, 0.0001),
    (45, 6, 55, 3, 0.0001),
    (46, 5, 65, 4, 0.0001),
    (47, 7, 25, 3, 0.0001),
]

summarised_city_manager_results = []
for i, test_arguments in enumerate(city_manager_runs):
    seed = test_arguments[0]
    max_depth = test_arguments[1]
    n_apartments = test_arguments[2]
    grid_length = test_arguments[3]
    epsilon = test_arguments[4]
    scip = build_and_optimise_city_manager(
        seed=seed,
        max_depth=max_depth,
        n_apartments=n_apartments,
        grid_length=grid_length,
        epsilon=epsilon,
        build_only=True,
    )
    scip.writeProblem(f"tests/data/instances/city_{i}.mps")
    summarised_city_manager_results.append(
        (i, scip.getStatus(), scip.getSolvingTime(), scip.getNTotalNodes())
    )

print(summarised_city_manager_results)

auto_manufacturer_runs = [
    (0, "gbdt", 10, 6, 10, 0.8),
    (1, "gbdt", 6, 6, 40, 0.8),
    (2, "gbdt", 6, 10, 20, 0.5),
    (3, "gbdt", 6, 6, 5, 0.9),
    (4, "gbdt", 5, 15, 40, 0.2),
    (5, "rf", 6, 6, 40, 0.8),
    (6, "rf", 6, 10, 40, 0.8),
    (7, "rf", 6, 10, 20, 0.5),
    (8, "rf", 6, 6, 10, 0.9),
    (9, "rf", 5, 15, 40, 0.2),
]

summarised_auto_results = []
for i, test_arguments in enumerate(auto_manufacturer_runs):
    seed = test_arguments[0]
    gbdt_or_rf = test_arguments[1]
    max_depth = test_arguments[2]
    n_estimators = test_arguments[3]
    min_fuel_efficiency = test_arguments[4]
    min_resale_ratio = test_arguments[5]
    scip = build_and_optimise_auto_manufacturer(
        seed=seed,
        gbdt_or_rf=gbdt_or_rf,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_fuel_efficiency=min_fuel_efficiency,
        min_resale_ratio=min_resale_ratio,
        build_only=True,
    )
    scip.writeProblem(f"tests/data/instances/auto_{i}.mps")
    summarised_auto_results.append(
        (i, scip.getStatus(), scip.getSolvingTime(), scip.getNTotalNodes())
    )

print(summarised_auto_results)

adversarial_example_runs = [
    (40, 16, (60, 20), 8000),
    (41, 16, (60, 40), 10000),
    (42, 16, (80, 40), 8000),
    (43, 16, (40, 20), 10000),
    (44, 32, (20, 10), 10000),
    (45, 32, (40, 20), 10000),
    (46, 32, (60, 40), 10000),
    (47, 32, (30, 30), 25000),
    (48, 32, (80, 40), 6000),
    (49, 64, (20, 10), 10000),
]

summarised_adversarial_results = []
for i, test_arguments in enumerate(adversarial_example_runs):
    seed = test_arguments[0]
    n_pixel_1d = test_arguments[1]
    layer_sizes = test_arguments[2]
    image_number = test_arguments[3]
    scip = build_and_optimise_adversarial_mnist_torch(
        seed=seed,
        n_pixel_1d=n_pixel_1d,
        layer_sizes=layer_sizes,
        image_number=image_number,
        test=False,
        build_only=True,
    )
    scip.writeProblem(f"tests/data/instances/adversarial_{i}.mps")
    summarised_adversarial_results.append(
        (i, scip.getStatus(), scip.getSolvingTime(), scip.getNTotalNodes())
    )

print(summarised_adversarial_results)
