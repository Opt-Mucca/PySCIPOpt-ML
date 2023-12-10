PySCIPOpt-ML
=====================

PySCIPOpt-ML is a python interface to automatically formulate Machine Learning (ML) models into Mixed-Integer Programs (MIPs).
PySCIPOPT-ML allows users to easily optimise MIPs with embedded ML constraints.

Installation
--------------------
``pyscipopt-ml`` can be installed from PyPI using ``pip``.
Python 3.8 or higher is required.

.. code-block:: bash

   pip install pyscipopt-ml

Helpful Information
====================

* Looking what ML models are supported? Try :doc:`Supported <./supported>`
* Looking for the MIP formulations of each model? Try :doc:`Mixed Integer Formulations <./formulations>`
* Interested in installation options? Try the README at the `GitHub Repo <https://github.com/Opt-Mucca/PySCIPOpt-ML/>`_
* Having trouble or a feature is missing? Raise an issue at `GitHub <https://github.com/Opt-Mucca/PySCIPOpt-ML/issues>`_
* Want to see the API? Try :doc:`API Reference Manual <./api-reference>`

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   basics

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   example_basic
   example_advanced

.. toctree::
   :maxdepth: 2
   :caption: Machine Learning Models
   :hidden:

   supported
   formulations

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api-reference
   bibliography
