Advanced Example - Wine Manufacturer
#####################################

Explanation and Description
===========================

In this example we take the point of view of a wine manufacturer.

Thanks to available open source data :cite:t:`wine`, which was accessed
`here <https://www.kaggle.com/datasets/piyushgoyal443/red-wine-dataset>`_, we can build
a ML predictor for wine quality given a set of attributes of the wine.
This can either be modelled as a regression model,
with the quality being in the range [0,1], or some quality threshold can be set,
and the task made into a classification task. For this exercise we model it is a regression task.

The goal of the MIP in the following example is to create a diverse bouquet of wines with the highest average
quality. To do this, we make an assumption that the features of the grapes and resulting wine
are identical. In this example we can purchase grapes from a set of suppliers, all of whom
have grapes that by themselves would result in low quality wine. However, we can blend the grapes
such that the predicted quality of the wine increases. For additional constraints, each supplier
has a limit on the amount of grapes that can be sold, and an associated cost per unit, where we have a total budget.
Finally, we assume that each wine of the created bouquet will have the same volume.


We describe the MIP formulation mathematically:

- Let :math:`I` be the index set of the wine bouquet that will be created.
- Let :math:`J` be the index set of features for each grape / wine.
- Let :math:`K` be the index set of vineyards where the grapes can be purchased

- Let :math:`x_{i,j}` be the value of feature j for wine i
- Let :math:`y_i` be the quality of the wine i
- Let :math:`m_{i,k}` be the amount of grapes from vineyard k used in wine i

- let :math:`f` be the ML predictor that determines the quality of a wine

- Let v[j][k] be the constant values of feature j from vineyard k
- Let c[k] be the constant cost of one unit of grapes from vineyard k
- Let a[k] be the amount of grapes available from vineyard k
- Let B be the total budget available

.. math::

    \begin{align*}
    &\text{max     } & \sum_i y_i & \\
    &s.t.   & x_{i,j} = \sum_k m_{i,k} * v[j][k] & \quad \forall i, j \\
    &  & \sum_k m_{i,k} = 1 & \quad \forall i \\
    &  & \sum_i m_{i,k} \leq a[k] & \quad \forall k \\
    &  & y_i = f(x_{i,1}, ..., x_{i, |J|}) & \quad \forall i \\
    &  & \sum_{i,k} m_{i,k} * c[k] \leq B
    \end{align*}


Code Walkthrough
=================

Below we will introduce a function for creating this example with a variety of ML frameworks
that can train gradient boosting decision trees and random forests.

This example is taken directly from one of the tests in the GitHub repository. See more
examples by searching there.

.. code-block:: python

    def build_and_optimise_wine_manufacturer(
        seed=42,
        n_vineyards=35,
        n_wines_to_produce=5,
        min_wine_quality=4.25,
        sklearn_xgboost_lightgbm="sklearn",
        gbdt_or_rf="rf",
        n_estimators=3,
        max_depth=3,
        epsilon=0.0001,
    ):
        assert sklearn_xgboost_lightgbm in ("sklearn", "xgboost", "lightgbm")
        assert gbdt_or_rf in ("gbdt", "rf")

        # Path to red wine data
        data_dict = read_csv_to_dict("./tests/data/wineQualityReds.csv")

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
        if sklearn_xgboost_lightgbm == "sklearn":
            if gbdt_or_rf == "rf":
                reg = RandomForestRegressor(
                    random_state=seed, n_estimators=n_estimators, max_depth=max_depth
                ).fit(X, quality)
            else:
                reg = GradientBoostingRegressor(
                    random_state=seed, n_estimators=n_estimators, max_depth=max_depth
                ).fit(X, quality)
        elif sklearn_xgboost_lightgbm == "xgboost":
            if gbdt_or_rf == "gbdt":
                reg = XGBRegressor(
                    random_state=seed, n_estimators=n_estimators, max_depth=max_depth
                ).fit(X, quality)
            else:
                reg = XGBRFRegressor(
                    random_state=seed, n_estimators=n_estimators, max_depth=max_depth
                ).fit(X, quality)
        else:
            if gbdt_or_rf == "gbdt":
                reg = LGBMRegressor(
                    random_state=seed, n_estimators=n_estimators, max_depth=max_depth
                ).fit(X, quality)
            else:
                reg = LGBMRegressor(
                    random_state=seed,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    boosting_type="rf",
                    bagging_freq=1,
                    bagging_fraction=0.5,
                ).fit(X, quality)

        # Create artificial data from some vineyards
        np.random.seed(seed)
        vineyard_order = np.arange(X.shape[0])
        np.random.shuffle(vineyard_order)
        vineyard_litre_limits = np.random.uniform(0.25, 0.35, n_vineyards)
        vineyard_costs = np.random.uniform(1, 2, n_vineyards)
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
            scip, reg, feature_vars, quality_vars, unique_naming_prefix="wine_", epsilon=epsilon
        )

        # Add a constraint ensuring minimum wine quality on those produced
        for i in range(n_wines_to_produce):
            scip.addCons(quality_vars[i][0] >= min_wine_quality, name=f"min_quality_{i}")

        # Set the SCIP objective
        scip.setObjective(
            quicksum(quality_vars[i][0] for i in range(n_wines_to_produce)) / n_wines_to_produce,
            sense="maximize"
        )

        return scip

Two important things to note in the insertion of the ML predictors.

- While it was a single function call to insert the ML predictor, the ML predictor was actually
  added n many times. Specifically, the same ML predictor was inserted for each of the generated wines

- When using decision trees, or any ML predictors that are based on decision trees, it is important
  to be aware of the epsilon value. In the above example we set a default value of 0.0001. This ensures
  that the output of the ML predictor matches the ML framework, but it removes a small portion of the
  feasible region, risking getting an infeasible model for a feasible problem. This is by default 0,
  because while the error can be arbitrarily large for decision trees, the error stems only from a
  numerically insignificant perturbation of the input

Back to the example: We can then create various models with different characteristics and different ML frameworks
by changing the parameters of the input. For example, to insert ML predictors from XGBoost using
Random Forests, we would do the following:

.. code-block:: python

    # Build the SCIP model with embedded ML predictors
    scip = build_and_optimise_wine_manufacturer(
        seed=42,
        n_vineyards=35,
        n_wines_to_produce=5,
        min_wine_quality=4.25,
        sklearn_xgboost_lightgbm="xgboost",
        gbdt_or_rf="rf",
        n_estimators=3,
        max_depth=3,
        epsilon=0.0001
        )

    # Optimise the SCIP model
    scip.optimize()

    # We can check the "error" of the MIP embedding via the difference between SKLearn and SCIP output
    if np.max(pred_cons.get_error()) > 10**-3:
        error = np.max(pred_cons.get_error())
        raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -3}")

    return scip

