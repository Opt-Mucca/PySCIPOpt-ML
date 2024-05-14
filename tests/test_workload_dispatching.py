import numpy as np
from pyscipopt import Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from utils import train_torch_neural_network

from src.pyscipopt_ml.add_predictor import add_predictor_constr

"""
In this scenario we take the point of view of a job scheduler. Specifically, we would like
to schedule jobs to individual compute cores, such that the minimum efficieny of any core
is maximised.

The model and data is thanks to:
@article{lombardi2017empirical,
  title={Empirical decision model learning},
  author={Lombardi, Michele and Milano, Michela and Bartolini, Andrea},
  journal={Artificial Intelligence},
  volume={244},
  pages={343--367},
  year={2017},
  publisher={Elsevier}
}
The data was accessed here: https://bitbucket.org/m_lombardi/eml-aij-2015-resources/src/master/training/

We have built a predictor that takes as input the average CPI (clocks per instruction or cycles per instruction)
of jobs assigned to a compute core, the average CPI of its neighbours, and the average CPI of cores far away.
The predictor outputs the efficiency of the core in how it handles the assigned jobs.

The goal of this MIP is to maximise the minimum efficiency over all cores. It achieves this by assigning
jobs to each compute core such that the predicted efficiency of the worst core is maximised.

Let I be the index set of jobs
Let K be the index set of compute cores
Let N[k] be some neighbourhood of other compute cores for compute core k
The variable x[i][k] defines if job i is assigned to compute core k. The variables are binary
cpi[i] is the clock per instructions of each job, and is input to the problem

The MIP model is as follows

min -y + 1

sum_k x[i][k] = 1 for all k
sum_i x[i][k] = |I| / |K| for all k
avg_cpi[k] = (|K| / |I|) sum_i cpi[i] * x[i][k] for all k
avg_neigh[k] = (1 / |N[k]|) * sum_k'_N[k] avg_cpi[k] for all k
avg_far[k] = (1 / (|K| - |N[k]|)) * sum_k'_N-N[k] avg_cpi[k] for all k
eff[k] = f(avg_cpi[k], avg_neigh[k], avg_far[k])
y <= eff[k] for all k
"""


def read_data_for_core(k):
    with open(f"tests/data/WDP/ts{k}.txt") as s:
        data_raw = s.readlines()

    x = np.zeros((len(data_raw) - 3, 3))
    y = np.zeros((len(data_raw) - 3,))

    for i, line in enumerate(data_raw[3:]):
        line = line.split("\n")[0].split(" ")
        assert len(line) == 5
        x[i][0] = float(line[0])
        x[i][1] = float(line[2])
        x[i][2] = float(line[3])
        y[i] = float(line[4])

    return x, y


def get_neighbours(k, n, m):
    neighbours = set()
    if k % m != 0:
        neighbours.add(k - 1)
    if (k + 1) % m != 0:
        neighbours.add(k + 1)
    if k - m >= 0:
        neighbours.add(k - m)
    if k + m <= n * m - 1:
        neighbours.add(k + m)

    not_neighbours = {i for i in range(n * m)}
    not_neighbours = not_neighbours - neighbours

    return sorted(list(neighbours)), sorted(list(not_neighbours))


def build_and_optimise_workload_dispatching(
    num_layers_or_estimators=4,
    layer_size_or_depth=3,
    nn_or_gbdt="gbdt",
    formulation="sos",
    framework="sklearn",
    training_seed=42,
    data_seed=42,
    build_only=False,
):
    n = 6
    m = 8
    n_jobs = 6 * n * m
    data_random_state = np.random.RandomState(data_seed)
    training_random_state = np.random.RandomState(training_seed)

    # Initialise the SCIP Model and all variables
    scip = Model()
    job_vars = np.zeros((n_jobs, n * m), dtype=object)
    avg_cpi_vars = np.zeros((n * m,), dtype=object)
    avg_neighbour_cpi_vars = np.zeros((n * m,), dtype=object)
    avg_far_cpi_vars = np.zeros((n * m,), dtype=object)
    efficiency_vars = np.zeros((n * m,), dtype=object)
    max_min_efficiency_var = scip.addVar(vtype="C", lb=0, ub=1, name="max_min_eff")

    for i in range(n * m):
        for j in range(n_jobs):
            job_vars[j][i] = scip.addVar(vtype="B", name=f"job_{j}_core_{i}")
        avg_cpi_vars[i] = scip.addVar(vtype="C", lb=-1, ub=0.5, name=f"avg_cpi_{i}")
        avg_neighbour_cpi_vars[i] = scip.addVar(vtype="C", lb=-1, ub=0.5, name=f"avg_n_cpi_{i}")
        avg_far_cpi_vars[i] = scip.addVar(vtype="C", lb=-1, ub=0.5, name=f"avg_f_cpi_{i}")
        efficiency_vars[i] = scip.addVar(vtype="C", lb=0, ub=1, name=f"eff_{i}")

    # Read the data and train + embed all ML predictors
    min_avg_cpi = np.inf
    max_avg_cpi = -np.inf
    predictor_constraints = []
    for i in range(n * m):
        x, y = read_data_for_core(i)
        min_avg_cpi = min(np.min(x[:, 0]), min_avg_cpi)
        max_avg_cpi = max(np.max(x[:, 0]), max_avg_cpi)
        layer_sizes = tuple([layer_size_or_depth for i in range(num_layers_or_estimators)])
        if nn_or_gbdt == "nn":
            if framework == "sklearn":
                reg = MLPRegressor(
                    hidden_layer_sizes=layer_sizes,
                    random_state=training_random_state,
                    learning_rate_init=0.01,
                ).fit(x, y)
            else:
                reg = train_torch_neural_network(
                    x, y, num_layers_or_estimators, layer_size_or_depth, training_seed
                )
        else:
            reg = GradientBoostingRegressor(
                n_estimators=num_layers_or_estimators,
                random_state=training_random_state,
                max_depth=layer_size_or_depth,
            ).fit(x, y)
        pred_input = [avg_cpi_vars[i], avg_neighbour_cpi_vars[i], avg_far_cpi_vars[i]]
        pred_cons = add_predictor_constr(
            scip,
            reg,
            pred_input,
            [efficiency_vars[i]],
            unique_naming_prefix=f"p_{i}_",
            formulation=formulation,
        )
        predictor_constraints.append(pred_cons)

    # Randomly generate an instance and create the appropriate constraints
    cpi = data_random_state.uniform(
        low=(min_avg_cpi - 0.01), high=(max_avg_cpi + 0.01), size=n_jobs
    )
    for i in range(n * m):
        # Add the average CPI constraint
        avg_cpi = sum(cpi[j] * job_vars[j][i] for j in range(n_jobs)) / 6
        scip.addCons(avg_cpi_vars[i] == avg_cpi, name=f"avg_cpi_cons_{i}")
        # Generate the neighbours of the core
        neighbours, not_neighbours = get_neighbours(i, n, m)
        # Add the average neighbour constraint
        neighbour_avg = sum(avg_cpi_vars[k] for k in neighbours) / (len(neighbours))
        scip.addCons(avg_neighbour_cpi_vars[i] == neighbour_avg, name=f"avg_cpi_neigh_cons_{i}")
        # Add the average far away core constraint
        not_neighbour_avg = sum(avg_cpi_vars[k] for k in not_neighbours) / len(not_neighbours)
        scip.addCons(avg_far_cpi_vars[i] == not_neighbour_avg, name=f"avg_far_cpi_cons_{i}")
        # Add the constraint that global efficiency is less than any single core efficiency
        scip.addCons(max_min_efficiency_var <= efficiency_vars[i], name=f"eff_bound_{i}")

    scip.setObjective(-max_min_efficiency_var + 1)

    if not build_only:
        scip.optimize()
        for pred_cons in predictor_constraints:
            if np.max(pred_cons.get_error()) > 10**-4:
                error = np.max(pred_cons.get_error())
                raise AssertionError(f"Max error {error} exceeds threshold of {10 ** -4}")

    return scip


def test_gbdt_workload_dispatching():
    build_and_optimise_workload_dispatching(
        nn_or_gbdt="gbdt", num_layers_or_estimators=4, layer_size_or_depth=2
    )


def test_nn_workload_dispatching():
    build_and_optimise_workload_dispatching(
        nn_or_gbdt="nn", num_layers_or_estimators=2, layer_size_or_depth=3
    )


def test_nn_workload_dispatching_bigm():
    build_and_optimise_workload_dispatching(
        nn_or_gbdt="nn", num_layers_or_estimators=2, layer_size_or_depth=3, formulation="bigm"
    )
