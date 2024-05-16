import numpy as np
from pyscipopt import Model, quicksum
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from utils import read_csv_to_dict, train_torch_neural_network

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we take the point of view of an agency tasked with replanting a stretch of land.

We have access to open source data (thanks to:
@article{wood2023tree,
  title={Tree seedling functional traits mediate plant-soil
  feedback survival responses across a gradient of light availability},
  author={Wood, Katherine EA and Kobe, Richard K and Ib{\'a}{\~n}ez, In{\'e}s and McCarthy-Neumann, Sarah},
  journal={Plos one},
  volume={18},
  number={11},
  pages={e0293906},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
Data accessed here: https://www.kaggle.com/datasets/yekenot/tree-survival-prediction)
for which we have built a predictor. The predictor takes as input a variety of features ranging
from soil properties to light properties, and determines the survival rate of the planted tree.

The goal of this MIP is to plant a set of trees such that some budget constraints are satisfied,
some diversity criteria on planted trees is satisfied, and the survival rate of the
planted trees is maximised. In the stretch of land where the trees are planted, the location
contains static features such as predicted sunlight, but the planter has the chance to sterilise sections of soil.
The planter must then decide in which location to plant which trees.

Let I, J be the index set of grid coordinates
Let T be the index set of the type of trees that can be planted
Let K be the index set of the features
Let x[i][j][k] be the value of feature k for the soil at grid point (i,j)
Let p[i][j][t] be the binary variable indicating iff tree t is planted at (i,j)
Let s[i][j]t[] be the survival rate of tree t if planted at grid point (i,j)
Let s'[i][j][t] be the adjusted survival rate (0 if not planted) of tree t at grid point (i,j)
Let min_survive[t] be the minimum amount of trees that need to survive of type t
Let costs[t] be the cost of planting tree t, and budget be the total budget
Let sterilisation_budget be the amount of grid points that be sterilised
Let f be the ML predictor for the survival of tree t at grid point i,j with soil feature x[i][j]

The MIP model is:

sum_t p[i][j][t] = 1 for all i,j
sum_{i,j} s'[i][j][t] >= min_survive[t] forall t
sum_t (sum_{i,j} p[i][j][t] * cost[t]) <= budget
sum_{i,j} x[i][j][sterilisation_idx] <= sterilisation_budget
s[i][j][t] = f(x[i][j][:])
p[i][j][t] = 1 => s'[i][j][t] <= s[i][j][t] forall i,j,t
p[i][j][t] = 0 => s'[i][j][t] <= 0

max(sum_{i,j,t} s'[i][j][t])
"""


def build_and_optimise_tree_planting(
    data_seed=42,
    training_seed=42,
    predictor_type="linear",
    formulation="sos",
    framework="sklearn",
    max_depth=5,
    n_estimators_layers=2,
    layer_size=8,
    n_grid_size=10,
    build_only=False,
):
    assert predictor_type in ("linear", "decision_tree", "gbdt", "mlp")
    training_random_state = np.random.RandomState(training_seed)
    data_random_state = np.random.RandomState(data_seed)
    max_sterilise = data_random_state.randint(low=8, high=13)
    costs = data_random_state.uniform(low=25, high=45, size=4)
    min_trees = data_random_state.uniform(
        low=n_grid_size**2 / 80, high=n_grid_size**2 / 60, size=4
    )
    max_budget = (n_grid_size**2) * np.median(costs) * 1.2
    # Read the csv data
    data_dict = read_csv_to_dict("./tests/data/tree_survivability.csv")

    # Extract the min and max light values
    light_values = np.array(
        [float(data_dict["Light_ISF"][i]) for i in range(len(data_dict["Light_ISF"]))]
    )
    min_light = np.min(light_values)
    max_light = np.max(light_values)

    # Now create the X and y data from the csv that is used to train the regression models
    species = sorted(list(set(data_dict["Species"])))
    species_idx_dict = {tree_type: i for i, tree_type in enumerate(species)}
    n_entries = len(data_dict["Light_ISF"])
    X = [[] for _ in range(len(species))]
    y = [[] for _ in range(len(species))]
    for i in range(n_entries):
        idx = species_idx_dict[data_dict["Species"][i]]
        X[idx].append([])
        if data_dict["Event"][i] == "1":
            y[idx] += [0]
        elif data_dict["Harvest"][i] == "X":
            y[idx] += [1]
        else:
            assert data_dict["Alive"][i] == "X", print(i)
            y[idx] += [1]
        X[idx][-1].append(light_values[i])
        X[idx][-1].append(data_dict["Light_Cat"][i] == "low")
        X[idx][-1].append(data_dict["Light_Cat"][i] == "med")
        X[idx][-1].append(data_dict["Light_Cat"][i] == "high")
        X[idx][-1].append(data_dict["Myco"][i] == "AMF")
        X[idx][-1].append(data_dict["SoilMyco"][i] == "AMF")
        X[idx][-1].append(data_dict["Sterile"][i] == "Sterile")

    # Create and train the regression model
    regression_models = []
    for i in range(len(species)):
        X_i = np.array(X[i])
        y_i = np.array(y[i])
        if predictor_type == "linear":
            reg = LinearRegression().fit(X_i, y_i)
        elif predictor_type == "decision_tree":
            reg = DecisionTreeRegressor(
                random_state=training_random_state, max_depth=max_depth
            ).fit(X_i, y_i)
        elif predictor_type == "gbdt":
            reg = GradientBoostingRegressor(
                random_state=training_random_state,
                n_estimators=n_estimators_layers,
                max_depth=max_depth,
            ).fit(X_i, y_i.reshape(-1))
        elif predictor_type == "mlp":
            if framework == "sklearn":
                hidden_layers = tuple([layer_size for i in range(n_estimators_layers)])
                reg = MLPRegressor(
                    random_state=training_random_state, hidden_layer_sizes=hidden_layers
                ).fit(X_i, y_i.reshape(-1))
            else:
                reg = train_torch_neural_network(
                    X_i, y_i, n_estimators_layers, layer_size, training_seed, reshape=True
                )
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        regression_models.append(reg)

    # Initialise the SCIP Model
    scip = Model()

    # Create variables deciding which tree is planted in which grid point and the survival rate of each planted tree
    tree_planting_vars = np.zeros((n_grid_size, n_grid_size, 4), dtype=object)
    tree_survive_vars = np.zeros((n_grid_size, n_grid_size, 4), dtype=object)
    tree_adjusted_survive_vars = np.zeros((n_grid_size, n_grid_size, 4), dtype=object)
    for i in range(n_grid_size):
        for j in range(n_grid_size):
            for k in range(4):
                tree_planting_vars[i][j][k] = scip.addVar(
                    vtype="B", name=f"plant_tree_{i}_{j}_{k}"
                )
                tree_survive_vars[i][j][k] = scip.addVar(
                    vtype="C", lb=-10, ub=10, name=f"survive_tree_{i}_{j}_{k}"
                )
                tree_adjusted_survive_vars[i][j][k] = scip.addVar(
                    vtype="C", lb=-10, ub=1, name=f"adjusted_survive_tree_{i}_{j}_{k}"
                )

    # Ensure that only a single tree is planted in a single grid point
    for i in range(n_grid_size):
        for j in range(n_grid_size):
            scip.addCons(
                quicksum(tree_planting_vars[i][j][k] for k in range(4)) == 1,
                name=f"square_{i}_{j}",
            )

    # Ensure an expected minimum amount of each tree type survives
    for k in range(4):
        scip.addCons(
            quicksum(
                quicksum(tree_adjusted_survive_vars[i][j][k] for j in range(n_grid_size))
                for i in range(n_grid_size)
            )
            >= min_trees[k],
            name=f"min_trees_{k}",
        )

    # Ensure that the budget is respected, and not more trees are planted than can be afforded
    scip.addCons(
        quicksum(
            quicksum(
                quicksum(costs[k] * tree_planting_vars[i][j][k] for k in range(4))
                for j in range(n_grid_size)
            )
            for i in range(n_grid_size)
        )
        <= max_budget,
        name=f"tree_budget",
    )

    # Ensure that a tree can only survive if it is planted
    for i in range(n_grid_size):
        for j in range(n_grid_size):
            for k in range(4):
                scip.addConsIndicator(
                    tree_adjusted_survive_vars[i][j][k] <= tree_survive_vars[i][j][k],
                    tree_planting_vars[i][j][k],
                    name=f"tree_survive_ind_{i}_{j}_{k}_0",
                )
                scip.addConsIndicator(
                    tree_adjusted_survive_vars[i][j][k] <= 0,
                    tree_planting_vars[i][j][k],
                    activeone=False,
                    name=f"tree_survive_ind_{i}_{j}_{k}_2",
                )

    # Create feature variables
    feature_variables = np.zeros((n_grid_size, n_grid_size, 7), dtype=object)
    for i in range(n_grid_size):
        light_isf = max_light - ((i / n_grid_size) * (max_light - min_light))
        for j in range(n_grid_size):
            light_cat = data_random_state.randint(0, 3)
            myco = data_random_state.randint(0, 2)
            soil_myco = data_random_state.randint(0, 2)
            for k in range(7):
                if k == 0:
                    feature_variables[i][j][k] = scip.addVar(
                        vtype="C", lb=min_light, ub=max_light, name=f"feature_{i}_{j}_{k}"
                    )
                    scip.fixVar(feature_variables[i][j][k], light_isf)
                else:
                    feature_variables[i][j][k] = scip.addVar(
                        vtype="B", name=f"feature_{i}_{j}_{k}"
                    )
                    if (
                        (k == 1 and light_cat == 0)
                        or (k == 2 and light_cat == 1)
                        or (k == 3 and light_cat == 2)
                        or (k == 4 and myco == 0)
                        or (k == 5 and soil_myco == 0)
                    ):
                        scip.fixVar(feature_variables[i][j][k], 1)
                    elif k == 6:
                        pass
                    else:
                        scip.fixVar(feature_variables[i][j][k], 0)

    # Create a constraint that only a certain amount of squares can be sterilised
    scip.addCons(
        quicksum(
            quicksum(feature_variables[i][j][6] for j in range(n_grid_size))
            for i in range(n_grid_size)
        )
        <= max_sterilise,
        name="sterilisation_budget",
    )

    # Add the predictors to the MIP
    pred_cons_list = []
    for i in range(len(species)):
        pred_cons_list.append(
            add_predictor_constr(
                scip,
                regression_models[i],
                feature_variables.reshape(-1, 7),
                tree_survive_vars[:, :, i].reshape(-1, 1),
                unique_naming_prefix=f"predictor_{i}_",
                epsilon=0.0001,
                formulation=formulation,
            )
        )

    # Set the objective to maximise amount of trees that survive
    scip.setObjective(
        -quicksum(
            quicksum(
                quicksum(tree_adjusted_survive_vars[i][j][k] for k in range(4))
                for j in range(n_grid_size)
            )
            for i in range(n_grid_size)
        )
        + n_grid_size**2
    )

    if not build_only:
        # Optimise the model
        scip.optimize()

        # We can check the "error" of the MIP embedding via the difference between SKLearn and SCIP output
        for pred_cons in pred_cons_list:
            if np.max(pred_cons.get_error()) > 10**-5:
                error = np.max(pred_cons.get_error())
                raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -5}")

    return scip


def test_tree_planting_linear():
    scip = build_and_optimise_tree_planting(
        data_seed=42,
        training_seed=42,
        predictor_type="linear",
        max_depth=5,
        n_estimators_layers=2,
        layer_size=8,
        n_grid_size=10,
    )


def test_tree_planting_decision_tree():
    scip = build_and_optimise_tree_planting(
        data_seed=42,
        training_seed=42,
        predictor_type="decision_tree",
        max_depth=5,
        n_estimators_layers=2,
        layer_size=8,
        n_grid_size=10,
    )


def test_tree_planing_gbdt():
    scip = build_and_optimise_tree_planting(
        data_seed=18,
        training_seed=35,
        predictor_type="gbdt",
        max_depth=5,
        n_estimators_layers=8,
        layer_size=8,
        n_grid_size=10,
    )


def test_tree_planting_mlp():
    scip = build_and_optimise_tree_planting(
        data_seed=18,
        training_seed=35,
        predictor_type="mlp",
        max_depth=5,
        n_estimators_layers=2,
        layer_size=6,
        n_grid_size=3,
    )


def test_tree_planting_mlp_bigm():
    scip = build_and_optimise_tree_planting(
        data_seed=18,
        training_seed=35,
        predictor_type="mlp",
        formulation="bigm",
        max_depth=5,
        n_estimators_layers=2,
        layer_size=6,
        n_grid_size=3,
    )
