"""Module for formulating a :external+sklearn:py:class:`sklearn.cluster.KMeans`
 into a PySCIPOpt Model.
"""

import numpy as np

from ..exceptions import NoModel
from ..modelling import AbstractPredictorConstr
from ..modelling.classification import argmax_bound_formulation
from ..modelling.var_utils import create_vars
from .skgetter import SKgetter


def add_centroid_cluster_constr(
    scip_model,
    centroid_clusteror,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    formulation="l2",
    **kwargs,
):
    """Formulate centroid_clusteror in scip_model.

    The formulation predicts the values of output_vars using input_vars according to
    centroid_clusteror.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the predictor should be inserted.
    centroid_clusteror : :external+sklearn:py:class:`sklearn.cluster.KMeans`
        The centroid clusteror to insert as predictor.
    input_vars : list or np.ndarray
        Decision variables used as input for centroid clustering in model.
    output_vars : list or np.ndarray, optional
        Decision variables used as output for centroid clustering in model.
    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.
    formulation : str, optional
        The formulation type used when embedding the centroid clustering predictor.
        Valid types are "l2" (standard norm, same as the predictor), and "l1" for a linearised version.
        Warning: The linearised version will incorrectly label some points.

    Returns
    -------
    CentroidClusterConstr
       Object containing information about what was added to scip_model to formulate
       centroid_clusteror.

    Note
    ----
    |VariablesDimensionsWarn|
    """

    return CentroidClusterConstr(
        scip_model,
        centroid_clusteror,
        input_vars,
        output_vars,
        unique_naming_prefix,
        formulation,
        **kwargs,
    )


class CentroidClusterConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained :external+sklearn:py:class:`sklearn.cluster.KMeans`
     with SCIP

    |ClassShort|.
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars,
        unique_naming_prefix,
        formulation,
        **kwargs,
    ):
        if hasattr(predictor, "n_clusters"):
            n_clusters = predictor.n_clusters
        else:
            n_clusters = predictor.cluster_centers_.shape[0]
        if n_clusters <= 1:
            raise NoModel(predictor, "Single cluster model is redundant")
        if n_clusters == 2:
            self.output_size = 1
        else:
            self.output_size = n_clusters
        if formulation not in ["l1", "l2"]:
            raise NoModel(predictor, f"Formulation {formulation} is invalid")
        self.formulation = formulation
        SKgetter.__init__(self, predictor, **kwargs)
        AbstractPredictorConstr.__init__(
            self, scip_model, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Predict output variables y from input variables X using the centroids.

        Both X and y should be arrays or lists of variables of conforming dimensions.
        """

        n_samples = self.input.shape[0]
        n_features = self.input.shape[-1]
        if hasattr(self.predictor, "n_clusters"):
            n_clusters = self.predictor.n_clusters
        else:
            n_clusters = self.predictor.cluster_centers_.shape[0]

        # Create additional variables for distance to eac cluster
        dist_vars = create_vars(
            self.scip_model,
            shape=(n_samples, n_clusters),
            vtype="C",
            lb=None,
            ub=None,
            name_prefix=self.unique_naming_prefix + "dist_",
        )

        # Create constraints that measure distance from each sample to each centroid
        dist_cons = np.zeros((n_samples, n_clusters), dtype=object)
        if self.formulation == "l1":
            l1_dist_vars = create_vars(
                self.scip_model,
                shape=(n_samples, n_clusters, n_features, 2),
                vtype="C",
                lb=0,
                name_prefix=self.unique_naming_prefix + "l1_dist_",
            )
            l1_dist_cons = np.zeros((n_samples, n_clusters, n_features), dtype=object)
            l1_sos_cons = np.zeros((n_samples, n_clusters, n_features), dtype=object)
        for j in range(n_clusters):
            centroid = self.predictor.cluster_centers_[j]
            for i in range(n_samples):
                if self.formulation == "l1":
                    for k in range(n_features):
                        name = self.unique_naming_prefix + f"l1_dist_{i}_{j}_{k}"
                        l1_dist_cons[i][j][k] = self.scip_model.addCons(
                            self.input[i][k] - centroid[k]
                            == l1_dist_vars[i][j][k][0] - l1_dist_vars[i][j][k][1],
                            name=name,
                        )
                        name = self.unique_naming_prefix + f"l1_sos_{i}_{j}_{k}"
                        l1_sos_cons[i][j][k] = self.scip_model.addConsSOS1(
                            [l1_dist_vars[i][j][k][0], l1_dist_vars[i][j][k][1]], name=name
                        )
                    sum_dist = sum(
                        l1_dist_vars[i][j][k][0] + l1_dist_vars[i][j][k][1]
                        for k in range(n_features)
                    )
                else:
                    sum_dist = sum(
                        (self.input[i][k] - centroid[k]) ** 2 for k in range(n_features)
                    )
                name = self.unique_naming_prefix + f"dist_cons_{i}_{j}"
                dist_cons[i][j] = self.scip_model.addCons(dist_vars[i][j] == sum_dist, name=name)

        # Add created variables and constraints
        if self.formulation == "l1":
            self._created_vars.append(l1_dist_vars)
            self._created_cons.append(l1_dist_cons)
            self._created_cons.append(l1_sos_cons)
        self._created_vars.append(dist_vars)
        self._created_cons.append(dist_cons)

        # Add argmax constraint for closest centroid (turn variables to negative for argmin)
        if n_clusters == 2:
            dist_bin_vars = create_vars(
                self.scip_model,
                shape=(n_samples, n_clusters),
                vtype="B",
                name_prefix=self.unique_naming_prefix + "max_dist_",
            )
            argmax_vars, argmax_cons = argmax_bound_formulation(
                self.scip_model, -dist_vars, dist_bin_vars, self.unique_naming_prefix
            )
            link_output_cons = np.zeros((n_samples, 1), dtype=object)
            for i in range(n_samples):
                name = self.unique_naming_prefix + f"link_max_out_{i}"
                link_output_cons[i][0] = self.scip_model.addCons(
                    self.output[i][0] == dist_bin_vars[i][1], name=name
                )
            self._created_vars.append(dist_bin_vars)
            self._created_cons.append(link_output_cons)
        else:
            argmax_vars, argmax_cons = argmax_bound_formulation(
                self.scip_model, -dist_vars, self.output, self.unique_naming_prefix
            )
        for new_vars in argmax_vars:
            self._created_vars.append(new_vars)
        for new_cons in argmax_cons:
            self._created_cons.append(new_cons)
