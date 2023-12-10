Mixed Integer Formulations
##########################

In this page, we give a quick overview of the mixed-integer formulations used to
represent the various machine learning (ML) models supported by the package.

Throughout,
we denote by :math:`x` the input to the ML model (i.e. the independent variables)
and by :math:`y` the output of the ML model (i.e. the dependent variables).

For all examples we will consider only a single sample as input. That sample
depending on the model will have multiple features and multiple outputs,
being either continuous (regression) or discrete (classification). Inputting more
samples, which can be done in all cases by this package, creates the same
set of constraints per sample.

Argmax Formulation
====================

These formulations are used in most cases of classification, i.e., when
the output needs to be categorical. Different formulations are used depending on
the dimension of the output :math:`y`. For all formulations of argmax there
are potential issues when multiple classes have approximately the largest value.
For these cases we will use :math:`\epsilon` to explain when issues can arise,
where by default in SCIP :math:`\epsilon = 10^{-6}`. Unlike standard argmax,
instead of returning the index of the largest argument, we want the index
of the largest argument to take value 1 and all other indices to take value 0.
We denote :math:`y'` as
the output of the ML model before argmax has been applied.

In the scenario where :math:`y` is single
dimensional (which normally makes an argmax redundant),
we interpret argmax as checking whether the value is greater than `0.5` or not.

.. math::

    &y \leq y' + 0.5

    &y \geq y' - 0.5

This formulation has an issue of tolerances around 0.5. So values in the range
:math:`[0.5 - \epsilon, 0.5 + \epsilon]` can all be either 0 or 1.

In the scenario where :math:`y` is two-dimensional, we formulate argmax with
indicator constraints

.. math::

    &y_0 = 1 \Rightarrow y'_0 \leq y'_1

    &y_1 = 1 \Rightarrow y'_1 \leq y'_0

    &y_0 + y_1 = 1

This formulation also has an issue with tolerances. If
both :math:`y'_0` and :math:`y'_1` take values differing by less than :math:`\epsilon`,
then either :math:`y_0` or :math:`y_1` can be selected by SCIP.

In the scenario where :math:`y` is three-dimensional or more, we formulate
argmax with SOS constraints. Let :math:`m \in \mathbb R` be an additional
variable used to find the maximum value,
and let :math:`s \in \mathbb{R}^{n}_{\geq 0}` be additional slack variables.

.. math::

    &y'_i + s_i + m = 0 \quad \forall i \in \{0,...,n-1\}

    &SOS1(s_i, y_i) \quad \forall i \in \{0,...,n-1\}

    &\sum_{i \in \{0,...,n-1\}} y_i = 1

This formulation has an issue with tolerances when there are classes with output values
less than :math:`\epsilon` smaller than the largest output value. In such a case SCIP can
select either the largest value label or any of those :math:`\epsilon` close class labels
as argmax.

For ease of notation, this constraint will simply be referred to as argmax(:math:`y'`)

Linear Regression
=================

Denoting by :math:`\beta \in \mathbb R^{n+1}` the computed weights of linear regression,
its model takes the form

.. math::

  y = \sum_{i=1}^n \beta_i x_i + \beta_0.

Since this is linear, it can be represented directly in SCIP using
linear constraints. Note that the model fits other techniques such as Ridge, Lasso,
and ElasticNet.

Logistic Regression
===================

The standard logistic function, also referred to as the sigmoid function in
some communities, is :math:`f(x) = \frac{1}{1 + e^{-x}}`.

In the case of regression, and a single dimensional output, the logistic
regression formulation is

.. math::

  y = f(\sum_{i=1}^n \beta_i x_i + \beta_0) = \frac{1}{1 + e^{- \sum_{i=1}^n
  \beta_i x_i - \beta_0}}

In the case of regression with multi-dimensional output, the regression
formulation depends on scikit-learn. This can change depending on user
defined parameters within the framework. The two potential formulations
are

.. math::

    y_j = \frac{f(\sum_{i=1}^n \beta_{i,j} x_i + \beta_{0,j})}{\sum_k f(\sum_{i=1}^n \beta_{i,k} x_i + \beta_{0,k}))} \quad \forall j

.. math::
    y_j = \frac{e^{\sum_{i=1}^n \beta_{i,j} x_i + \beta_{0,j}}}{\sum_k e^{\sum_{i=1}^n \beta_{i,k} x_i + \beta_{0,k}}} \quad \forall j

In the case of classification we avoid modelling the non-linearities. For :math:`y` being single
dimension we formulate logistic regression for classification with

.. math::

    &y \leq 1 \sum_{i=1}^n \beta_i x_i + \beta_0

    &y \geq \sum_{i=1}^n \beta_i x_i + \beta_0

    &y \in \{0, 1\}

In the case of multi-class classification we formulate logistic regression using the argmax
formulation

.. math::

    y'_j =& \sum_{i=1}^n \beta_{i,j} x_i + \beta_{0,j} \quad \forall j

    y =& argmax(y')


Neural Networks
===============

The package currently models dense neural network with ReLU, Sigmoid, and Tanh activations.
For all formulations we let i be the node index of the input layer and j be node index
of the out layer.

For dense layers with a ReLU activation function, we introduce slack variables
:math:`s \in \mathbb{R}^{n}_{\geq 0}`, with the formulation of the layer given by:

.. math::

    y_j = \sum_{i=1}^n \beta_{i,j} x_i + \beta_{0,j} + s_j \quad &\forall j

    SOS1(y_j, s_j) \quad &\forall j

Note that this formulation is non-standard in the literature. The standard formulation
uses big-M constraints, for which bounds are found through feasibility and optimality based
bound tightening procedures. Empirically such formulations have been shown to
be the current state-of-the-art. Such a formulation fails completely, however,
when the big-M values becomes sufficiently large, and is less friendly
numerically overall w.r.t. the difference between the true output of the predictor
and that which SCIP returns. Therefore we have decided to use SOS1 constraints,
which will likely be slower on well-scaled neural networks.

For dense layers with a Sigmoid activation function the formulation is:

.. math::

    y_j = \frac{1}{1 + e^{-(\sum_{i=1}^n \beta_{i,j} x_i + \beta_{0,j})}} \quad \forall j

For dense layers with a Tanh activation function the formulation is:

.. math::

    y_j = \frac{1 - e^{-2(\sum_{i=1}^n \beta_{i,j} x_i + \beta_{0,j})}}{1 + e^{-2(\sum_{i=1}^n \beta_{i,j} x_i + \beta_{0,j})}} \quad \forall j

As the maximum is preserved over all these activation functions, and other activation functions
such as Softmax, the inserted predictor constraint for classification purposes does not explicitly
model the final activation layer. In such a case the formulation used is:

.. math::

    y'_j =& argmax(\sum_{i=1}^n \beta_{i,j} x_i + \beta_{0,j}) \quad \forall j

    y =& argmax(y')


Decision Tree
========================

In a decision tree, each leaf :math:`l` is defined by a number of constraints
on the input features of the tree that correspond to the branches taken in the
path leading to :math:`l`. We formulate decision trees by introducing one
binary decision variable :math:`\delta_l` for each leaf of the tree.

In the decision tree exactly one leaf is chosen. This constraint is formulated as:

.. math::
   \sum_{l} \delta_l = 1

To ensure that the input vector maps to the correct leaf, however, we need to introduce
additional notation and constraints.
For a node :math:`v`, we denote by :math:`i_v` the
feature used for splitting and by :math:`\theta_v` the value at which the split
is made. At a leaf :math:`l` of the tree, we have a set :math:`\mathcal L_l` of inequalities of
the form :math:`x_{i_v} \le \theta_v` corresponding to the left branches leading to
:math:`l` and a set :math:`\mathcal R_l` of inequalities of
the form :math:`x_{i_v} > \theta_v` corresponding to the right branches.

For each leaf, the inequalities describing :math:`\mathcal L_l` and :math:`\mathcal R_l`
are imposed using indicator constraints:

.. math::

   & \delta_l = 1 \rightarrow x_{i_v} \leq \theta_v - \frac{\epsilon}{2}, & & \forall x_{i_v} \le \theta_v \in \mathcal L_l,

   & \delta_l = 1 \rightarrow x_{i_v} \geq \theta_v + \frac{\epsilon}{2}, & & \forall x_{i_v} > \theta_v \in \mathcal R_l.

In our implementation, :math:`\epsilon` can be specified by a keyword parameter `epsilon` in
functions that add a decision tree constraint. By default the value for :math:`\epsilon` is 0.
When :math:`\epsilon` is smaller than the default tolerance in SCIP (as it is by default),
and you have a solution where :math:`x_{i_v} \approx \theta_v`, then SCIP can select an
arbitrary child node of that decision in the tree.

Here is a concrete example.
Let an internal node of the decision tree be for the feature :math:`x_4` and value 5.
Then the decisions are:

.. math::

    x_4 \leq 5

    x_4 \geq 5

When :math:`x_4 \approx 5`, both these conditions are true for SCIP, and therefore both child nodes
can be reached. The result is that for a value of :math:`x_4 = 4.999999999`, SCIP
could say that :math:`x_4 \geq 5`, and then the output of SCIP can be drastically
different to that which is returned by `decision_tree.predict()`.
The purpose of :math:`\epsilon` is to break these ties, and enforce
that only one of the decision can ever be true. The downside is that it introduces
a small area of model infeasibility. For instance, if :math:`\epsilon = 0.001`,
and the only solution to the above example is :math:`x_4 = 5`, then that
solution is no longer valid according to the formulation. Therefore,
we warn users to be careful when setting :math:`\epsilon` to be non-zero.

When using decision trees for classification, we create constraints
that ensure the correct class is selected depending on the leaf node. Let
:math:`y_j` be the output for class j, and :math:`L_j` be the set of leaf nodes
that predict class j. The constraint ensuring the class is selected
according to the leaf node is:

.. math::

    y_j = \sum_{l \in L_j} l


Random Forests
========================

The formulation of Random Forests is a linear combination (aggregation) of decision trees.
Each decision tree is represented using the model above. The same difficulties
with the choice of :math:`\epsilon` apply to this case.

In the case of classification, after the linear combination (aggregation) is performed,
the output is piped through the argmax formulation.

Gradient Boosting Trees
============================

The formulation of Gradient Boosting Trees is a linear combination (aggregation) of decision trees.
Each decision tree is represented using the model above. The same difficulties with
the choice of :math:`\epsilon` apply to this case.

In the case of classification, after the linear combination (aggregation) is performed,
the output is piped through the argmax formulation.