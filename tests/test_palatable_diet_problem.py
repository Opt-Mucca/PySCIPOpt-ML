import numpy as np
import pandas as pd
from pyscipopt import Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from utils import train_torch_neural_network

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we take the point of view of the World Food Programme (WFP). We imagine a simplified
variant of their food delivery problem, where one must minimise costs while ensuring that an
appropriate amount of food is delivered. Appropriate is defined both by some
nutritional value threshold and by some palatable threshold.

The model is a subproblem of the larger optimisation problem by:
@article{peters2021nutritious,
  title={The nutritious supply chain: optimizing humanitarian food assistance},
  author={Peters, Koen and Silva, S{\'e}rgio
  and Gon{\c{c}}alves, Rui and Kavelj, Mirjana and Fleuren, Hein
  and den Hertog, Dick and Ergun, Ozlem and Freeman, Mallory},
  journal={INFORMS Journal on Optimization},
  volume={3},
  number={2},
  pages={200--226},
  year={2021},
  publisher={INFORMS}
}
The data is accessed thanks to:
@article{maragno2023mixed,
  title={Mixed-integer optimization with constraint learning},
  author={Maragno, Donato and Wiberg, Holly and
  Bertsimas, Dimitris and Birbil, {\c{S}} {\.I}lker and den Hertog, Dick and Fajemisin, Adejuyigbe O},
  journal={Operations Research},
  year={2023},
  publisher={INFORMS}
}
Accessed here: https://github.com/hwiberg/OptiCL/tree/main/notebooks/WFP/processed-data

We have built a predictor that takes as input a variety of features (in this case the
amount of each type of food), and determines how palatable of the collection of food.

The goal of this MIP is to minimise the transport cost of food to those in need while ensuring that
all nutritional requirements are met and that the general collection of food
is palatable.

Let I be the index set of the type of food
Let J be the index set of the type of nutrient
Let x[i] be the variables representing the amount of fod of type i that is purchased
Let c[i] be the transport cost of food type i
Let n[i][j] be the nutritional value of nutrient j for one unit of food type i
Let N[j] be the minimum nutritional requirement for nutrient j
Let t be a value in range [0,1] (0: horrible, 1: incredible)
Let f be the ML predictor for how palatable a set of food is

The MIP model is:

min sum_i c[i] * x[i]

sum_i n[i][j] * x[i] >= N[j] forall j
x_salt = 5
x_sugar = 20
f(x) >= t
"""


def build_and_optimise_palatable_diet(
    mlp_gbdt_svm="mlp",
    formulation="sos",
    framework="sklearn",
    n_estimators_or_layers=2,
    layer_sizes_or_depth=16,
    degree=2,
    training_seed=42,
    data_seed=42,
    build_only=False,
):
    # Initialise a random state
    data_random_state = np.random.RandomState(data_seed)
    training_random_state = np.random.RandomState(training_seed)

    # Read in all the data
    nutr_val = pd.read_excel(
        "tests/data/WFP/instance_data.xlsx",
        sheet_name="nutr_val",
        index_col="Food",
        engine="openpyxl",
    ).to_numpy()
    nutr_req = pd.read_excel(
        "tests/data/WFP/instance_data.xlsx",
        sheet_name="nutr_req",
        index_col="Type",
        engine="openpyxl",
    ).to_numpy()
    cost_p = (
        pd.read_excel(
            "tests/data/WFP/instance_data.xlsx",
            sheet_name="FoodCost",
            index_col="Supplier",
            engine="openpyxl",
        )
        .iloc[0, :]
        .to_numpy()
    )
    n_food = nutr_val.shape[0]
    n_nutrients = nutr_val.shape[1]
    data = pd.read_csv("tests/data/WFP/palatable_data.csv").sample(frac=1)
    y = data["label"].to_numpy()
    x = data.drop(["label"], axis=1, inplace=False).to_numpy()

    # Train the regression model on whether the diet is palatable or not
    if mlp_gbdt_svm == "mlp":
        if framework == "sklearn":
            hidden_layer_sizes = tuple(
                [layer_sizes_or_depth for i in range(n_estimators_or_layers)]
            )
            reg = MLPRegressor(
                random_state=training_random_state,
                hidden_layer_sizes=hidden_layer_sizes,
            ).fit(x, y.reshape(-1))
        else:
            reg = train_torch_neural_network(
                x, y, n_estimators_or_layers, layer_sizes_or_depth, training_seed, reshape=True
            )
    elif mlp_gbdt_svm == "gbdt":
        reg = GradientBoostingRegressor(
            max_depth=5, n_estimators=n_estimators_or_layers, random_state=training_random_state
        ).fit(x, y)
    elif mlp_gbdt_svm == "svm":
        reg = SVR(kernel="poly", degree=degree).fit(x, y)
    else:
        raise ValueError(f"No known model for type {mlp_gbdt_svm}")

    # Create the SCIP Model and model all variables
    scip = Model()
    input_vars = np.zeros((1, n_food), dtype=object)
    for i in range(n_food):
        input_vars[0][i] = scip.addVar(vtype="C", lb=0, ub=10, name=f"x_{i}")
    output_vars = np.zeros((1, 1), dtype=object)
    output_vars[0][0] = scip.addVar(vtype="C", lb=0, ub=1, name="y")

    # Add the nutritional constraints
    for j in range(n_nutrients):
        scip.addCons(
            sum(nutr_val[i][j] * input_vars[0][i] for i in range(n_food)) >= nutr_req[0][j],
            name=f"nutr_req_{j}",
        )
    # Fix salt and sugar as they are always constant in the training data
    scip.addCons(input_vars[0][9] == 0.05, name="salt_const")
    scip.addCons(input_vars[0][20] == 0.2, name="sugar_const")

    # Set the objective to minimise costs
    scip.setObjective(sum(cost_p[i] * input_vars[0][i] for i in range(n_food)), sense="minimize")

    # Insert the ML predictor
    pred_cons = add_predictor_constr(
        scip,
        reg,
        input_vars,
        output_vars,
        unique_naming_prefix="reg_",
        epsilon=0.001,
        formulation=formulation,
    )

    # Add a minimum palatable constraint
    min_palatable = data_random_state.uniform(low=0.5, high=0.52)
    scip.addCons(output_vars[0][0] >= min_palatable, name="palatable")

    # Optimise the model!
    if not build_only:
        scip.optimize()
        if np.max(pred_cons.get_error()) > 10**-4:
            error = np.max(pred_cons.get_error())
            raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -4}")

    return scip


def test_svm_palatable_diet():
    build_and_optimise_palatable_diet(mlp_gbdt_svm="svm", degree=1)


def test_nn_palatable_diet():
    build_and_optimise_palatable_diet(
        mlp_gbdt_svm="mlp", n_estimators_or_layers=2, layer_sizes_or_depth=16
    )


def test_nn_palatable_diet_bigm():
    build_and_optimise_palatable_diet(
        mlp_gbdt_svm="mlp", n_estimators_or_layers=2, layer_sizes_or_depth=16, formulation="bigm"
    )


def test_gbdt_palatable_diet():
    build_and_optimise_palatable_diet(
        mlp_gbdt_svm="gbdt", n_estimators_or_layers=5, layer_sizes_or_depth=4
    )
