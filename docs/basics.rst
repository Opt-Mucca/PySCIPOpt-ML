Basics
######

Introduction
************

PySCIPOpt-ML is a Python package for the automatic formulation of machine learning (ML) models
into Mixed-Integer Programs (MIPs) using `SCIP <https://github.com/scipopt/scip>`_.
This automatic formulation allows users to easily optimise mathematical optimisation problems with embedded
ML constraints, without worrying about the exact formulation and how to extract the data from the ML interface.

The package currently supports various `scikit-learn
<https://scikit-learn.org/stable/>`_ objects. It can also embed
gradient boosting regression models from `XGBoost <https://xgboost.readthedocs.io/en/stable/>`_, and
`LightGBM <https://lightgbm.readthedocs.io/en/stable/>`_. Finally, it supports Sequential Neural Networks from
`PyTorch <https://pytorch.org/docs/master/>`_.

The package is actively developed and users are encouraged to raise an issue on
`GitHub <https://github.com/Opt-Mucca/PySCIPOpt-ML/issues>`_ if there are ML
models that are currently available, or there are optimisation related problems with the created MIPs.

Install
*******

We encourage to install the package via pip (or add it to your
`requirements.txt` file):


.. code-block:: console

  (venv) pip install pyscipopt-ml


.. note::

  If not already installed, this should install the :pypi:`pyscipopt` and :pypi:`numpy`
  packages.

.. note::

  The package can also be installed from source. To do so first clone the package and then run:

  .. code-block:: bash

    python -m pip install .

.. note::

  The following table lists the version of the relevant packages that are
  tested and supported.

  .. list-table::
    :widths: 50
    :align: center
    :header-rows: 1

    * - Package
    * - :pypi:`pyscipopt`
    * - :pypi:`numpy`
    * - :pypi:`torch`
    * - :pypi:`scikit-learn`
    * - :pypi:`lightgbm`
    * - :pypi:`xgboost`

  Installing any of the machine learning packages is only required if the
  predictor you want to insert uses them (i.e. to insert a Scikit-Learn based predictor
  you need to have :pypi:`scikit-learn` installed).


Usage
*****

The main function provided by the package is
:py:func:`pyscipopt_ml.add_predictor_constr`. It takes as arguments: a PySCIPOpt Model, a
:doc:`supported ML model <supported>`, input PySCIPOpt variables, and
output PySCIPOpt variables.

By calling the function, the PySCIPOpt Model is augmented with variables and
constraints so that, in a solution, the values of the output variables are
predicted by the regression model from the values of the input variables. More
formally, if we denote by :math:`g` the prediction function of the embedded ML
model, by :math:`x` the input variables and by :math:`y` the output variables,
then :math:`y = g(x)` in any solution.

The function :py:func:`add_predictor_constr <pyscipopt_ml.add_predictor_constr>`
returns a modeling object derived from the class
:py:class:`AbstractPredictorConstr
<pyscipopt_ml.modeling.AbstractPredictorConstr>`. That object keeps track of all
the variables and constraints that have been added to the PySCIPOpt to
establish the relationship between input and output variables of the ML model.

The modeling object can perform a few tasks:

   * It can print a summary of what it added with the :py:meth:`print_stats
     <pyscipopt_ml.modelling.AbstractPredictorConstr.print_stats>` method.
   * Once SCIP computed a solution to the optimization problem, it can compute
     the difference between what the ML model predicts from the input
     values and the values of the output variables in SCIP's solution with the
     :py:meth:`get_error
     <pyscipopt_ml.modelling.AbstractPredictorConstr.get_error>` method.


The function :py:func:`add_predictor_constr <pyscipopt_ml.add_predictor_constr>` is
a shorthand that should add the correct model for any supported ML
model, but individual functions for each ML model are also available.
For the list of frameworks and ML models supported, and the corresponding
functions please refer to the :doc:`supported <supported>` section. We also briefly
outline how the various ML models are formulated in SCIP in the :doc:`Mixed Integer Formulations <formulations>`
section.

For examples on how to use the package please refer to the the :doc:`example <example_basic>`.
