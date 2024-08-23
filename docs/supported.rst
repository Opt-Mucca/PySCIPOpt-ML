Supported ML Models
###########################

The package currently supports various `Scikit-Learn
<https://scikit-learn.org/stable/>`_ objects. It can also embed
gradient boosting regression models from `XGBoost <https://xgboost.readthedocs.io/en/stable/>`_, and
`LightGBM <https://lightgbm.readthedocs.io/en/stable/>`_. Finally, it supports Sequential Neural Networks from
`PyTorch <https://pytorch.org/docs/master/>`_ and `Keras <https://keras.io/api/>`_.
In :doc:`Mixed Integer Formulations <./formulations>`, we briefly outline the
MIP formulations used for the various ML models.


Scikit-learn
------------
The following table lists the name of the models supported, the name of the
corresponding object in the Python framework, and the function that can be used
to insert it in a SCIP model.

.. list-table:: Supported ML models of :external+sklearn:std:doc:`scikit-learn <user_guide>`
   :widths: 25 25 50
   :header-rows: 1

   * - Machine Learning Model
     - Scikit-learn object
     - Functions to insert
   * - Ordinary Least Square
     - :external:py:class:`LinearRegression
       <sklearn.linear_model.LinearRegression>`
       :external:py:class:`Ridge
       <sklearn.linear_model.Ridge>`
       :external:py:class:`ElasticNet
       <sklearn.linear_model.ElasticNet>`
       :external:py:class:`Lasso
       <sklearn.linear_model.Lasso>`
     - :py:mod:`add_linear_regression_constr
       <pyscipopt_ml.sklearn.linear_regression>`
   * - Partial Least Square
     - :external:py:class:`PLSRegression
       <sklearn.cross_decomposition.PLSRegression>`
       :external:py:class:`PLSRegression
       <sklearn.cross_decomposition.PLSCanonical>`
     - :py:mod:`add_pls_regression_constr
       <pyscipopt_ml.sklearn.pls>`
   * - Logistic Regression
     - :external:py:class:`LogisticRegression
       <sklearn.linear_model.LogisticRegression>`
     - :py:mod:`add_logistic_regression_constr
       <pyscipopt_ml.sklearn.logistic_regression>`
   * - Neural-network
     - :external:py:class:`MLPRegressor
       <sklearn.neural_network.MLPRegressor>`
       :external:py:class:`MLPClassifier
       <sklearn.neural_network.MLPClassifier>`
     - :py:mod:`add_mlp_regressor_constr
       <pyscipopt_ml.sklearn.mlp>`
       :py:mod:`add_mlp_classifier_constr
       <pyscipopt_ml.sklearn.mlp>`
   * - Decision tree
     - :external:py:class:`DecisionTreeRegressor
       <sklearn.tree.DecisionTreeRegressor>`
       :external:py:class:`DecisionTreeClassifier
       <sklearn.tree.DecisionTreeClassifier>`
     - :py:mod:`add_decision_tree_regressor_constr
       <pyscipopt_ml.sklearn.decision_tree>`
       :py:mod:`add_decision_tree_classifier_constr
       <pyscipopt_ml.sklearn.decision_tree>`
   * - Gradient boosting
     - :external:py:class:`GradientBoostingRegressor
       <sklearn.ensemble.GradientBoostingRegressor>`
       :external:py:class:`GradientBoostingClassifier
       <sklearn.ensemble.GradientBoostingClassifier>`
     - :py:mod:`add_gradient_boosting_regressor_constr
       <pyscipopt_ml.sklearn.gradient_boosting>`
       :py:mod:`add_gradient_boosting_classifier_constr
       <pyscipopt_ml.sklearn.gradient_boosting>`
   * - Random Forest
     - :external:py:class:`RandomForestRegressor
       <sklearn.ensemble.RandomForestRegressor>`
       :external:py:class:`RandomForestClassifier
       <sklearn.ensemble.RandomForestClassifier>`
     - :py:mod:`add_random_forest_regressor_constr
       <pyscipopt_ml.sklearn.random_forest>`
       :py:mod:`add_random_forest_classifier_constr
       <pyscipopt_ml.sklearn.random_forest>`
   * - Support Vector Machines
     - :external:py:class:`SVR
       <sklearn.svm.SVR>`
       :external:py:class:`SVC
       <sklearn.svm.SVC>`
       :external:py:class:`LinearSVR
       <sklearn.svm.LinearSVR>`
       :external:py:class:`LinearSVC
       <sklearn.svm.LinearSVC>`
     - :py:mod:`add_support_vector_regressor_constr
       <pyscipopt_ml.sklearn.support_vector>`
       :py:mod:`add_support_vector_classifier_constr
       <pyscipopt_ml.sklearn.support_vector>`
   * - Centroid Clustering
     - :external:py:class:`KMeans
       <sklearn.cluster.KMeans>`
       :external:py:class:`MiniBatchKMeans
       <sklearn.cluster.MiniBatchKMeans>`
     - :py:mod:`add_centroid_cluster_constr
       <pyscipopt_ml.sklearn.centroid_cluster>`
   * - Pipeline
     - :external:py:class:`Pipeline
       <sklearn.pipeline.Pipeline>`
     - :py:mod:`add_pipeline_constr
       <pyscipopt_ml.sklearn.pipeline>`
   * - MultiOutput
     - :external:py:class:`MultiOutputClassifier
       <sklearn.multioutput.MultiOutputClassifier>`
       :external:py:class:`MultiOutputRegressor
       <sklearn.multioutput.MultiOutputRegressor>`
     - :py:mod:`add_multi_output_classifier_constr
       <pyscipopt_ml.sklearn.multi_output>`
       :py:mod:`add_multi_output_regressor_constr
       <pyscipopt_ml.sklearn.multi_output>`

PyTorch
-------

In PyTorch, only :external+torch:py:class:`torch.nn.Sequential` objects are
supported.

They can be embedded in a SCIP model with the function
:py:func:`pyscipopt_ml.torch.add_sequential_constr`.

Currently, only five types of layers are supported:

   * :external+torch:py:class:`Linear layers <torch.nn.Linear>`,
   * :external+torch:py:class:`ReLU layers <torch.nn.ReLU>`,
   * :external+torch:py:class:`Sigmoid layers <torch.nn.Sigmoid>`,
   * :external+torch:py:class:`Tanh layers <torch.nn.Tanh>`
   * :external+torch:py:class:`Softmax layers <torch.nn.Softmax>`
   * :external+torch:py:class:`Softplus layers <torch.nn.Softplus>`

In the case of the final layer being an activation function used for classification, e.g.
:external+torch:py:class:`Softmax <torch.nn.Softmax>`, simply set
`output_type=="classification"` when inserting the predictor constraint.
The result is that the class with highest value
is assigned value 1 and all other classes are assigned value 0. Essentially, explicitly modelling
the final activation function for classification purposes is unnecessary from a MIP perspective as
the maximum value is preserved after the function is applied.

Keras
------

For Keras, only `keras.Model <https://keras.io/api/models/model/>`_ and
`keras.Sequential <https://keras.io/api/models/sequential/>`_ are supported.

They can be embedded in a SCIP model with the function
:py:func:`pyscipopt_ml.keras.add_keras_constr`.

The supported layer types and activation functions are the same as in torch (see above).
This support holds for the classification case when the final layer is an unsupported activation function,
e.g. softmax. Please read the above explanation in the PyTorch section, and in such use cases set
`output_type="classification"` when inserting the predictor constraint.

ONNX
----

For ONNX we also support standard feed forward neural networks. These must be
provided in the `ModelProto <https://onnx.ai/onnx/api/classes.html#modelproto>`_ format.

They can be embedded in a SCIP model with the function
:py:func:`pyscipopt_ml.onnx.add_onnx_constr`.

The supported layer types and activation functions are the same as in torch and keras (see above).
The classification trick for the final layer is not done for ONNX models, so be warned that there will
be a performance difference for imported classification models.


XGBoost
-------

Models for XGBoost's Scikit-Learn interface can be embedded in a SCIP model.
The following table lists the name of the models supported, the name of the
corresponding object in the Python framework, and the function that can be used
to insert it in a SCIP model.

.. list-table:: Supported ML models of :external+xgb:std:doc:`xgboost <python/sklearn_estimator>`
   :widths: 25 50
   :header-rows: 1

   * - XGBoost object
     - Function to insert
   * - :external+xgb:py:class:`xgboost.XGBRegressor <xgboost.XGBRegressor>`
     - :py:mod:`add_xgbregressor_constr
       <pyscipopt_ml.xgboost.add_xgbregressor_constr>`
   * - :external+xgb:py:class:`xgboost.XGBClassifier <xgboost.XGBClassifier>`
     - :py:mod:`add_xgbclassifier_constr
       <pyscipopt_ml.xgboost.add_xgbclassifier_constr>`
   * - :external+xgb:py:class:`xgboost.XGBRFRegressor <xgboost.XGBRFRegressor>`
     - :py:mod:`add_xgbregressor_rf_constr
       <pyscipopt_ml.xgboost.add_xgbregressor_rf_constr>`
   * - :external+xgb:py:class:`xgboost.XGBRFClassifier <xgboost.XGBRFClassifier>`
     - :py:mod:`add_xgbclassifier_rf_constr
       <pyscipopt_ml.xgboost.add_xgbclassifier_rf_constr>`

Currently only "gbtree" boosters are supported.

LightGBM
--------

Models for LightGBM's Scikit-Learn interface can be embedded in a SCIP model.
The following table lists the name of the models supported, the name of the
corresponding object in the Python framework, and the function that can be used
to insert it in a SCIP model.

.. list-table:: Supported ML models of :external+lgb:std:doc:`lightgbm <Python-API>`
   :widths: 25 50
   :header-rows: 1

   * - LightGBM object
     - Function to insert
   * - :external+lgb:py:class:`lightgbm.LGBMRegressor <lightgbm.LGBMRegressor>`
     - :py:mod:`add_lgbregressor_constr
       <pyscipopt_ml.lightgbm.add_lgbregressor_constr>`
   * - :external+lgb:py:class:`lightgbm.LGBMClassifier <lightgbm.LGBMClassifier>`
     - :py:mod:`add_lgbclassifier_constr
       <pyscipopt_ml.lightgbm.add_lgbclassifier_constr>`

Currently "gbdt" and "rf" boosters are supported.
