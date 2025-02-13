from test_adversarial_example import build_and_optimise_adversarial_mnist_torch
from test_auto_manufacturer import build_and_optimise_auto_manufacturer
from test_city_manager import build_and_optimise_city_manager
from test_palatable_diet_problem import build_and_optimise_palatable_diet
from test_simple_function_approximation import (
    build_and_optimise_function_approximation_model,
)
from test_tree_planting import build_and_optimise_tree_planting
from test_water_potability import build_and_optimise_water_potability
from test_wine_manufacturer import build_and_optimise_wine_manufacturer
from test_workload_dispatching import build_and_optimise_workload_dispatching

write_only = True
time_limit = 1200
generate_wine = False
generate_water = False
generate_auto = False
generate_tree = False
generate_function = False
generate_workload = False
generate_city = False
generate_diet = False
generate_adversarial = False

for d_s in [0, 1]:
    for t_s in [0, 1]:

        # Generate the wine instances
        wine_data = []
        for n_v in [30, 35]:
            for n_w in [4, 5]:
                # Generate the neural network instances
                for formulation in ["bigm", "sos"]:
                    for n_e, d in [(2, 16), (2, 32), (3, 16)]:
                        if generate_wine:
                            scip = build_and_optimise_wine_manufacturer(
                                data_seed=d_s,
                                training_seed=t_s,
                                n_vineyards=n_v,
                                n_wines_to_produce=n_w,
                                formulation=formulation,
                                framework="torch",
                                gbdt_rf_or_mlp="mlp",
                                n_estimators_layers=n_e,
                                layer_size=d,
                                epsilon=0.0001,
                                build_only=True,
                            )
                            if write_only:
                                scip.writeProblem(
                                    f"tests/surrogatelib/wine_{n_v}_{n_w}_mlp-{formulation}_{n_e}_{d}_torch_{d_s}_{t_s}.mps"
                                )
                            else:
                                scip.setParam("limits/time", time_limit)
                                scip.optimize()
                                wine_data.append(
                                    [
                                        n_v,
                                        n_w,
                                        formulation,
                                        n_e,
                                        d,
                                        scip.getStatus(),
                                        scip.getSolvingTime(),
                                        scip.getNTotalNodes(),
                                    ]
                                )
                # Generate the ensemble tree instances
                for fw in ["sklearn", "lightgbm", "xgboost"]:
                    for p_t in ["gbdt", "rf"]:
                        for n_e, d in [(4, 4), (3, 5)]:
                            if generate_wine:
                                scip = build_and_optimise_wine_manufacturer(
                                    data_seed=d_s,
                                    training_seed=t_s,
                                    n_vineyards=n_v,
                                    n_wines_to_produce=n_w,
                                    framework=fw,
                                    gbdt_rf_or_mlp=p_t,
                                    n_estimators_layers=n_e,
                                    max_depth=d,
                                    epsilon=0.0001,
                                    build_only=True,
                                )
                                if write_only:
                                    if fw == "sklearn":
                                        fw_short = "sk"
                                    elif fw == "lightgbm":
                                        fw_short = "lgb"
                                    else:
                                        fw_short = "xgb"
                                    scip.writeProblem(
                                        f"tests/surrogatelib/wine_{n_v}_{n_w}_{p_t}_{n_e}_{d}_{fw_short}_{d_s}_{t_s}.mps"
                                    )
                                else:
                                    scip.setParam("limits/time", time_limit)
                                    scip.optimize()
                                    wine_data.append(
                                        [
                                            n_v,
                                            n_w,
                                            fw,
                                            p_t,
                                            n_e,
                                            d,
                                            scip.getStatus(),
                                            scip.getSolvingTime(),
                                            scip.getNTotalNodes(),
                                        ]
                                    )

        # Generate the water instances
        water_data = []
        for n_w in [20, 25, 30]:
            # Generate the ensemble tree instances
            for n_e, d in [(6, 5), (7, 5), (8, 5)]:
                if generate_water:
                    scip = build_and_optimise_water_potability(
                        data_seed=d_s,
                        training_seed=t_s,
                        predictor_type="gbdt",
                        n_water_samples=n_w,
                        max_depth=d,
                        n_estimators_layers=n_e,
                        framework="sklearn",
                        build_only=True,
                    )
                    if write_only:
                        scip.writeProblem(
                            f"tests/surrogatelib/water_{n_w}_gbdt_{n_e}_{d}_sk_{d_s}_{t_s}.mps"
                        )
                    else:
                        scip.setParam("limits/time", time_limit)
                        scip.optimize()
                        water_data.append(
                            [
                                n_w,
                                "gbdt",
                                n_e,
                                d,
                                scip.getStatus(),
                                scip.getSolvingTime(),
                                scip.getNTotalNodes(),
                            ]
                        )
            # Generate the neural network instances
            for formulation in ["bigm", "sos"]:
                for n_e, d in [(7, 16), (8, 16)]:
                    if generate_water:
                        scip = build_and_optimise_water_potability(
                            data_seed=d_s,
                            training_seed=t_s,
                            predictor_type="mlp",
                            formulation=formulation,
                            n_water_samples=n_w,
                            layer_size=d,
                            n_estimators_layers=n_e,
                            framework="torch",
                            build_only=True,
                        )
                        if write_only:
                            scip.writeProblem(
                                f"tests/surrogatelib/water_{n_w}_mlp-{formulation}_{n_e}_{d}_torch_{d_s}_{t_s}.mps"
                            )
                        else:
                            scip.setParam("limits/time", time_limit)
                            scip.optimize()
                            water_data.append(
                                [
                                    n_w,
                                    "mlp",
                                    "torch",
                                    n_e,
                                    d,
                                    scip.getStatus(),
                                    scip.getSolvingTime(),
                                    scip.getNTotalNodes(),
                                ]
                            )

        # Generate the tree planting instances
        tree_data = []
        for g_s in [6, 7, 8]:
            for p_t in ["linear", "gbdt", "decision_tree"]:
                if p_t == "linear":
                    gbdt_list = [(1, 1)]
                elif p_t == "decision_tree":
                    gbdt_list = [(1, 4), (1, 5)]
                else:
                    gbdt_list = [(5, 4), (6, 4), (5, 5), (6, 5)]
                for n_e, d in gbdt_list:
                    if generate_tree:
                        scip = build_and_optimise_tree_planting(
                            data_seed=d_s,
                            training_seed=t_s,
                            predictor_type=p_t,
                            framework="sklearn",
                            max_depth=d,
                            n_estimators_layers=n_e,
                            build_only=True,
                        )
                        if write_only:
                            if p_t == "decision_tree":
                                p_t_short = "dt"
                            else:
                                p_t_short = p_t
                            if p_t == "gbdt":
                                scip.writeProblem(
                                    f"tests/surrogatelib/tree_{g_s}_{p_t_short}_{n_e}_{d}_sk_{d_s}_{t_s}.mps"
                                )
                            if p_t == "decision_tree":
                                scip.writeProblem(
                                    f"tests/surrogatelib/tree_{g_s}_{p_t_short}_{d}_sk_{d_s}_{t_s}.mps"
                                )
                            else:
                                scip.writeProblem(
                                    f"tests/surrogatelib/tree_{g_s}_{p_t_short}_sk_{d_s}_{t_s}.mps"
                                )
                        else:
                            scip.setParam("limits/time", time_limit)
                            scip.optimize()
                            tree_data.append(
                                [
                                    g_s,
                                    p_t,
                                    n_e,
                                    d,
                                    scip.getStatus(),
                                    scip.getSolvingTime(),
                                    scip.getNTotalNodes(),
                                ]
                            )

        # Generate function approximation instances
        function_data = []
        for n_i in [5, 6, 7]:
            for fw in ["sklearn", "keras", "torch"]:
                for formulation in ["bigm", "sos"]:
                    for n_e, d in [(3, 16), (3, 32), (4, 16), (5, 16)]:
                        if generate_function:
                            scip = build_and_optimise_function_approximation_model(
                                data_seed=d_s,
                                training_seed=t_s,
                                n_inputs=n_i,
                                framework=fw,
                                formulation=formulation,
                                layer_size=d,
                                n_layers=n_e,
                                build_only=True,
                            )
                            if write_only:
                                scip.writeProblem(
                                    f"tests/surrogatelib/function_{n_i}_mlp-{formulation}_{n_e}_{d}_{fw}_{d_s}_{t_s}.mps"
                                )
                            else:
                                scip.setParam("limits/time", time_limit)
                                scip.optimize()
                                function_data.append(
                                    [
                                        n_i,
                                        fw,
                                        formulation,
                                        n_e,
                                        d,
                                        scip.getStatus(),
                                        scip.getSolvingTime(),
                                        scip.getNTotalNodes(),
                                    ]
                                )

        # Generate city planning instances
        city_data = []
        for n_a in [25, 35, 45]:
            for p_t in ["dt", "gbdt"]:
                if p_t == "dt":
                    gbdt_list = [(1, 6), (1, 7), (1, 8)]
                else:
                    gbdt_list = [(2, 4), (2, 5), (3, 4), (3, 5)]
                for n_e, d in gbdt_list:
                    if generate_city:
                        scip = build_and_optimise_city_manager(
                            data_seed=d_s,
                            training_seed=t_s,
                            dt_gbdt_or_mlp=p_t,
                            framework="sklearn",
                            max_depth_or_layer_size=d,
                            n_estimators_layers=n_e,
                            n_apartments=n_a,
                            epsilon=0.0001,
                            build_only=True,
                        )
                        if write_only:
                            if p_t == "dt":
                                scip.writeProblem(
                                    f"tests/surrogatelib/city_{n_a}_{p_t}_{d}_sk_{d_s}_{t_s}.mps"
                                )
                            else:
                                scip.writeProblem(
                                    f"tests/surrogatelib/city_{n_a}_{p_t}_{n_e}_{d}_sk_{d_s}_{t_s}.mps"
                                )
                        else:
                            scip.setParam("limits/time", time_limit)
                            scip.optimize()
                            city_data.append(
                                [
                                    n_a,
                                    p_t,
                                    n_e,
                                    d,
                                    scip.getStatus(),
                                    scip.getSolvingTime(),
                                    scip.getNTotalNodes(),
                                ]
                            )

        # Generate auto manufacturer instances
        auto_data = []
        # The neural network instances first
        for formulation in ["bigm", "sos"]:
            for n_e, d in [(1, 64), (2, 16), (2, 32), (3, 16)]:
                if generate_auto:
                    scip = build_and_optimise_auto_manufacturer(
                        training_seed=t_s,
                        data_seed=d_s,
                        gbdt_rf_or_mlp="mlp",
                        formulation=formulation,
                        framework="torch",
                        max_depth_or_layer_size=d,
                        n_estimators_or_layers=n_e,
                        build_only=True,
                    )
                    if write_only:
                        scip.writeProblem(
                            f"tests/surrogatelib/auto_mlp-{formulation}_{n_e}_{d}_torch_{d_s}_{t_s}.mps"
                        )
                    else:
                        scip.setParam("limits/time", time_limit)
                        scip.optimize()
                        auto_data.append(
                            [
                                formulation,
                                "torch",
                                n_e,
                                d,
                                scip.getStatus(),
                                scip.getSolvingTime(),
                                scip.getNTotalNodes(),
                            ]
                        )
        # The ensemble tree instances next
        for p_t in ["gbdt", "rf"]:
            for n_e, d in [(8, 7), (7, 8), (8, 8)]:
                if generate_auto:
                    scip = build_and_optimise_auto_manufacturer(
                        training_seed=t_s,
                        data_seed=d_s,
                        gbdt_rf_or_mlp=p_t,
                        framework="sklearn",
                        max_depth_or_layer_size=d,
                        n_estimators_or_layers=n_e,
                        build_only=True,
                    )
                    if write_only:
                        scip.writeProblem(
                            f"tests/surrogatelib/auto_{p_t}_{n_e}_{d}_sk_{d_s}_{t_s}.mps"
                        )
                    else:
                        scip.setParam("limits/time", time_limit)
                        scip.optimize()
                        auto_data.append(
                            [
                                p_t,
                                "sklearn",
                                n_e,
                                d,
                                scip.getStatus(),
                                scip.getSolvingTime(),
                                scip.getNTotalNodes(),
                            ]
                        )

        # Generate adversarial attack instances
        adversarial_data = []
        for n_p in [14, 16, 18]:
            for formulation in ["bigm", "sos"]:
                for n_e, d in [(3, 16), (3, 32), (4, 16), (4, 32)]:
                    if generate_adversarial:
                        scip = build_and_optimise_adversarial_mnist_torch(
                            data_seed=d_s,
                            training_seed=t_s,
                            n_pixel_1d=n_p,
                            layer_size=d,
                            n_layers=n_e,
                            test=False,
                            formulation=formulation,
                            build_only=True,
                        )
                        if write_only:
                            scip.writeProblem(
                                f"tests/surrogatelib/adversarial_{n_p}_mlp-{formulation}_{n_e}_{d}_torch_{d_s}_{t_s}.mps"
                            )
                        else:
                            scip.setParam("limits/time", time_limit)
                            scip.optimize()
                            adversarial_data.append(
                                [
                                    n_p,
                                    formulation,
                                    "torch",
                                    n_e,
                                    d,
                                    scip.getStatus(),
                                    scip.getSolvingTime(),
                                    scip.getNTotalNodes(),
                                ]
                            )

        # Generate palatable diet problem instances
        palatable_data = []
        # Ensemble tree instances
        for n_e, d in [(10, 6), (11, 6), (12, 6)]:
            if generate_diet:
                scip = build_and_optimise_palatable_diet(
                    training_seed=t_s,
                    data_seed=d_s,
                    framework="sklearn",
                    n_estimators_or_layers=n_e,
                    layer_sizes_or_depth=d,
                    mlp_gbdt_svm="gbdt",
                    build_only=True,
                )
                if write_only:
                    scip.writeProblem(
                        f"tests/surrogatelib/palatable_gbdt_{n_e}_{d}_sk_{d_s}_{t_s}.mps"
                    )
                else:
                    scip.setParam("limits/time", time_limit)
                    scip.optimize()
                    palatable_data.append(
                        [
                            "gbdt",
                            "sklearn",
                            n_e,
                            d,
                            scip.getStatus(),
                            scip.getSolvingTime(),
                            scip.getNTotalNodes(),
                        ]
                    )

        # Generate workload dispatching instances
        workload_data = []
        for n_e, d in [(13, 3), (14, 3), (15, 3)]:
            if generate_workload:
                scip = build_and_optimise_workload_dispatching(
                    training_seed=t_s,
                    data_seed=d_s,
                    framework="sklearn",
                    nn_or_gbdt="gbdt",
                    num_layers_or_estimators=n_e,
                    layer_size_or_depth=d,
                    build_only=True,
                )
                if write_only:
                    scip.writeProblem(
                        f"tests/surrogatelib/workload_gbdt_{n_e}_{d}_sk_{d_s}_{t_s}.mps"
                    )
                else:
                    scip.setParam("limits/time", time_limit)
                    scip.optimize()
                    workload_data.append(
                        [
                            "gbdt",
                            "sklearn",
                            n_e,
                            d,
                            scip.getStatus(),
                            scip.getSolvingTime(),
                            scip.getNTotalNodes(),
                        ]
                    )

if not write_only:
    for generator_param, generator_data, generator_key in [
        (generate_wine, wine_data, "wine"),
        (generate_water, water_data, "water"),
        (generate_workload, workload_data, "workload"),
        (generate_function, function_data, "function"),
        (generate_diet, palatable_data, "diet"),
        (generate_adversarial, adversarial_data, "adversarial"),
        (generate_auto, auto_data, "auto"),
        (generate_tree, tree_data, "tree"),
        (generate_city, city_data, "city"),
    ]:
        if generator_param:
            print(f"{generator_key}: {generator_data}", flush=True)
