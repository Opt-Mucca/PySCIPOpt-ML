"""Module for formulating :external+onnx:py:class:`onnx.ModelProto` model in a PySCIPOPT Model.

Thanks to https://github.com/cog-imperial/OMLT/blob/main/src/omlt/io/onnx_parser.py for help with the parser

Notes to help with reading:
- This code is more general than necessary. It has the general structure (still missing parts) to later allow not
just fully connected layers. Many assert statements would need to be removed and code would need to be thoroughly
checked before introducing such features.
- A "node" here is used in the ONNX content, and can represent an entire layer.
"""

import numpy as np
import onnxruntime as ort
from onnx import numpy_helper

from ..exceptions import NoModel, NoSolution, ParameterError
from ..modelling.classification import argmax_bound_formulation
from ..modelling.neuralnet import AbstractNNLayer, BaseNNConstr
from ..modelling.var_utils import create_vars


def add_onnx_constr(
    scip_model,
    onnx_model,
    input_vars,
    output_vars=None,
    unique_naming_prefix="",
    output_type="regression",
    **kwargs,
):
    """Formulate an onnx ModelProto into scip_model.

    Parameters
    ----------
    scip_model : PySCIPOpt Model
        The SCIP model where the ONNX ModelProto should be inserted.

    onnx_model : :external+onnx:py:class:`onnx.ModelProto`
        The onnx model to insert as predictor.

    input_vars : np.ndarray
        Decision variables used as input for the feed forward neural network in PySCIPOpt Model.

    output_vars : np.ndarray
        Decision variables used as output for the feed forward neural network in the PySCIPOpt Model

    unique_naming_prefix : str, optional
        A unique naming prefix that is used before all variable and constraint names. This parameter is important if
        the SCIP model is later printed to file and many predictors are added to the same SCIP model.

    output_type : {"classification", "regression"}, default="regression"
        If the option chosen is "classification" the output is 1 for exactly one class and
        0 for all others. If the option chosen is "regression" then the output for
        each node of the final layer is the value from the ONNX model.

    Returns
    -------
    ONNXConstr
        Object containing information about what was added to model to insert the
        predictor into it

    Raises
    ------
    NoModel
        If a section of the ONNX model structure (layer or activation) is not implemented.

    Warning
    -------
    Only MatMul, Gemm, Relu, Sigmoid, Tanh, Softmax, and Softplus operator types are supported.
    Add, Cast, and Reshape are also often possible, although not in arbitrary contexts.

    Note
    ----
    |VariablesDimensionsWarn|

    """
    return ONNXConstr(
        scip_model,
        onnx_model,
        input_vars,
        output_vars,
        unique_naming_prefix,
        output_type,
        **kwargs,
    )


class ONNXConstr(BaseNNConstr):
    """
    Transform an ONNX ModelProto that represents a fully connected feed forward neural network
     to SCIP constraints with input and output as matrices of variables.

    |ClassShort|.
    """

    def __init__(
        self,
        scip_model,
        predictor,
        input_vars,
        output_vars=None,
        unique_naming_prefix="",
        output_type="regression",
        **kwargs,
    ):
        if output_type not in ["regression", "classification"]:
            raise NoModel(
                predictor, f"Output type {output_type} is neither regression nor classification"
            )
        self.output_type = output_type
        self._graph = predictor.graph

        # Check that the graph from ONNX contains only nodes that we can work with
        for node in self._graph.node:
            if node.op_type == "MatMul":
                if len(node.input) != 2:
                    raise ValueError(
                        f"MatMul node {node.name} input has {len(node.input)} many dimensions, not 2."
                    )
            elif node.op_type == "Gemm":
                if len(node.input) != 3:
                    raise ValueError(
                        f"Gemm node {node.name} input has {len(node.input)} many dimensions, not 3."
                    )
            elif node.op_type not in [
                "Add",
                "Relu",
                "Sigmoid",
                "Tanh",
                "Softmax",
                "Softplus",
                "Reshape",
                "Cast",
            ]:
                raise NoModel(predictor, f"ONNX node op type {node.op_type} not supported")

        # Verify the input and output dimensions
        if len(self._graph.input) != 1:
            raise NoModel(
                predictor,
                f"There are {len(self._graph.input)} many input tensors. Currently only one input vector"
                f" is supported",
            )
        if len(self._graph.output) != 1:
            raise NoModel(
                predictor,
                f"There are {len(self._graph.output)} many output tensors. Currently only one output vector"
                f" is supported",
            )

        # The initializers contain all constant data, e.g., weights and biases (is often input to a node)
        self._initializers = {}
        for initializer in self._graph.initializer:
            self._initializers[initializer.name] = numpy_helper.to_array(initializer)

        # Derive the input size of the model. Do this by flattening along all dimensions
        # Simultaneously populate the nodes and node_by_output dictionaries
        # The nodes dictionary contains tuples "input" or "node", ONNX node, list of nodes it feeds into)
        # The nodes_by_output dictionary contains a mapping from an output name to a node name
        # An output in this context is likely going to be represented by a set of created variables
        # Note: In the case of the node being an original input node the output of the node maps to itself
        self.input_size = None
        self._nodes_by_output = {}
        self._nodes = {}
        for onnx_input in self._graph.input:
            for dim in onnx_input.type.tensor_type.shape.dim:
                dim_value = dim.dim_value
                if dim_value > 0:
                    self._nodes_by_output[onnx_input.name] = onnx_input.name
                    self._nodes[onnx_input.name] = ("input", onnx_input.type, [])
                    if self.input_size is None:
                        self.input_size = dim_value
                    else:
                        self.input_size *= dim_value
        # Further populate the nodes and nodes_by_output dictionaries
        for node in self._graph.node:
            for output in node.output:
                self._nodes_by_output[output] = node.name
        for node in self._graph.node:
            # Add node not connected to anything
            self._nodes[node.name] = ("node", node, [])

            # Map inputs by their output name
            node_inputs = [
                self._nodes_by_output[onnx_input]
                for onnx_input in node.input
                if onnx_input not in self._initializers
            ]

            if len(node_inputs) > 0:
                # Now connect inputs to the current node
                for onnx_input in node_inputs:
                    self._nodes[onnx_input][2].append(node.name)

        # Determine the output size
        # This is non-trivial..... It cannot be taken from the output of the graph
        # We therefore have to work backwards and find the final matrix multiplication or addition
        self.output_size = None
        node = self._graph.node[-1]
        max_depth = len(self._graph.node)
        i = 0
        while i <= max_depth:
            # If the node has no dimensional information then find a previous node that feeds into it
            if node.op_type not in ["Add", "MatMul", "Gemm"]:
                if node.name not in self._nodes_by_output:
                    for node_input_name in node.input:
                        if node_input_name in self._nodes_by_output:
                            node = self._nodes[self._nodes_by_output[node_input_name]][1]
                else:
                    node = self._nodes[self._nodes_by_output[node.name]][1]
                i += 1
                continue
            if node.op_type == "Add":
                for onnx_input in node.input:
                    if onnx_input in self._initializers:
                        self.output_size = self._initializers[onnx_input].shape[-1]
                        break
                if self.output_size is not None:
                    break
            elif node.op_type == "MatMul":
                [_, matrix_name] = list(node.input)
                self.output_size = self._initializers[matrix_name].shape[-1]
                break
            elif node.op_type == "Gemm":
                attr = _collect_attributes(node)
                [_, _, bias_name] = list(node.input)
                self.output_size = self._initializers[bias_name].shape[-1]
                break
            i += 1
            node = self._nodes[self._nodes_by_output[node.name]][1]
        if self.output_size is None:
            self.output_size = self.input_size

        super().__init__(
            scip_model, predictor, input_vars, output_vars, unique_naming_prefix, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Add the predictor constraints to SCIP."""

        # Initialise the node input to layer map. This is used to get the input variables for later layers
        self._node_layer_map = {}
        for onnx_input in self._graph.input:
            self._node_layer_map[onnx_input.name] = self

        # Traverse the graph. This begins by populating the node stack with the input nodes
        input_nodes = set()
        for onnx_input in self._graph.input:
            input_nodes.add(onnx_input.name)
        self._node_stack = list(input_nodes)
        # Keep track of the number of processed nodes for naming purposes
        self.num_processed_nodes = 0
        # Keep visiting a new node until the stack has been completely explored. As the ONNX graph is directed and
        # acyclic this will always terminate
        while len(self._node_stack) > 0:
            node_name = self._node_stack.pop()
            type_, node, next_nodes = self._nodes[node_name]
            # Skip cases where node is input
            if type_ == "node":
                layer = self._visit_node(node, next_nodes)
            else:
                for node in next_nodes:
                    self._node_stack.append(node)

        # In the case of classification force the output to be binary with the argmax formulation
        n_samples = self.input.shape[0]
        if len(self._layers) > 0:
            input_vars = self._layers[-1].output
            activation = self._layers[-1].activation
        else:
            input_vars = self.input
            activation = "identity"
        if self.output_type == "classification":
            if activation == "sigmoid":
                one_dim_center = 0
            else:
                one_dim_center = 0.5
            new_vars, new_cons = argmax_bound_formulation(
                self.scip_model,
                input_vars,
                self.output,
                self.unique_naming_prefix,
                one_dim_center=one_dim_center,
            )
            for var_set in new_vars:
                self._created_vars.append(var_set)
            for cons_set in new_cons:
                self._created_cons.append(cons_set)
        else:
            # In the case of regression we need to tie the global output to the output of the final layer
            if input_vars.shape[-1] != self.output_size:
                raise ValueError(
                    f"Dimension of final layer != dimension of output vars. "
                    f"{input_vars.shape[-1]} != {self.output_size}"
                )
            match_output_cons = np.zeros(self.output.shape, dtype=object)
            for i in range(n_samples):
                for j in range(input_vars.shape[-1]):
                    name = self.unique_naming_prefix + f"match_out_{i}_{j}"
                    match_output_cons[i][j] = self.scip_model.addCons(
                        input_vars[i][j] == self.output[i][j], name=name
                    )
            self._created_cons.append(match_output_cons)

    def _visit_node(self, node, next_nodes):
        """
        Function for exploring / processing / parsing / consuming the current node

        Parameters
        ----------
        node : onnx.NodeProto
            The node that is currently being explored
        next_nodes : list
            The nodes next in line to be explored (output nodes of the current node)

        Returns
        -------
        layer : AbstractNNLayer or None
            The layer of the NN generated by the current node

        """
        # Separate the handling of node types by case
        if node.op_type == "MatMul":
            next_nodes, layer = self._consume_dense_nodes(node, next_nodes)
        elif node.op_type == "Gemm":
            next_nodes, layer = self._consume_gemm_dense_nodes(node, next_nodes)
        elif node.op_type == "Reshape" or node.op_type == "Cast":
            for node_input in node.input:
                if node_input not in self._initializers:
                    for node_output in node.output:
                        self._node_layer_map[node_output] = self._node_layer_map[node_input]
            layer = None
        else:
            raise Exception(f"Unhandled node type {node.op_type}")

        for node in next_nodes:
            self._node_stack.append(node)

        return layer

    def _consume_dense_nodes(self, node, next_nodes):
        """Starting from a MatMul node parse nodes to form a dense Ax + b node."""
        if node.op_type != "MatMul":
            raise ValueError(f"{node.name} is op type {node.op_type}, not MatMul.")

        # Extract input to the node
        [name_of_node_input_for_layer, matrix_name] = list(node.input)
        node_weights = self._initializers[matrix_name]

        if len(next_nodes) != 1:
            raise ValueError(
                f"Next nodes must have length 1, {next_nodes} has length {len(next_nodes)}"
            )

        # Expect 'Add' node ahead
        type_, node, next_nodes = self._nodes[next_nodes[0]]
        if type_ != "node":
            raise TypeError(f"Expected a node next, got a {type_} instead.")
        if node.op_type != "Add":
            raise ValueError(
                f"The first node after the MatMul node, {node.name}, is a {node.op_type} node, not Add."
            )

        # Extract biases
        [bias_name_1, bias_name_2] = list(node.input)
        if bias_name_1 in self._initializers:
            node_biases = self._initializers[bias_name_1]
        elif bias_name_2 in self._initializers:
            node_biases = self._initializers[bias_name_2]
        else:
            raise ValueError(f"Node inputs were not found in graph initializers.")

        if len(node_weights.shape) != 2:
            raise ValueError(f"Node weights must be a 2-dimensional matrix.")
        if node_weights.shape[1] != node_biases.shape[1]:
            raise ValueError(
                f"Dimension mismatch between weights {node_weights.shape} and biases {node_biases.shape}"
            )
        if len(node.output) != 1:
            raise ValueError(
                f"Node output is {node.output} but should consist of a single ONNX node."
            )

        # Now model the actual layer
        next_nodes, layer = self._add_dense_layer_matmul_gemm(
            self._node_layer_map[name_of_node_input_for_layer],
            node_weights,
            node_biases,
            node,
            node.name,
            next_nodes,
        )

        return next_nodes, layer

    def _consume_gemm_dense_nodes(self, node, next_nodes):
        """Starting from a Gemm node parse nodes to form a dense aAB + bC node."""
        if node.op_type != "Gemm":
            raise ValueError(f"{node.name} is op type {node.op_type}, not Gemm.")

        # Extract input to the node
        attr = _collect_attributes(node)
        if "alpha" in attr:
            alpha = attr["alpha"]
        else:
            alpha = 1.0
        if "beta" in attr:
            beta = attr["beta"]
        else:
            beta = 1.0
        [name_of_node_input_for_layer, matrix_name, bias_name] = list(node.input)
        node_weights = self._initializers[matrix_name]
        # Transpose B
        if "transB" in attr and attr["transB"] == 1:
            node_weights = np.transpose(node_weights)
        node_biases = self._initializers[bias_name]
        # Apply the appropriate scalars
        node_weights = node_weights * alpha
        node_biases = beta * node_biases

        # Now model the actual layer
        next_nodes, layer = self._add_dense_layer_matmul_gemm(
            self._node_layer_map[name_of_node_input_for_layer],
            node_weights,
            node_biases,
            node,
            name_of_node_input_for_layer,
            next_nodes,
        )

        return next_nodes, layer

    def _add_dense_layer_matmul_gemm(
        self, input_layer, weights, biases, node, node_name, next_nodes
    ):
        """
        Add a dense layer to the SCIP model using information extracted from a matmul or gemm node

        Parameters
        ----------
        input_layer : ONNXConstr or AbstractNNLayer
            The input layer whose output will input to the now created layer
        weights : np.ndarray
            The weights (coefficients) of the layer
        biases : np.ndarray
            The biases of the layer
        node : onnx.NodeProto
            The current node being considered (can be different from node_name due to matmul -> add etc)
        node_name : str
            The name of the node the layer is being constructed from
        next_nodes : list
            The list of nodes that should be explored after the current node is processed

        Returns
        -------
        next_nodes : list
            The list of nodes that should be explored after the current node is processed
        layer : AbstractNNLayer
            The created dense layer

        """

        # Determine the activation function of the layer (if one exists)
        activation = "identity"
        if isinstance(input_layer, ONNXConstr):
            input_vars = input_layer.input
        elif isinstance(input_layer, AbstractNNLayer):
            input_vars = input_layer.output
        else:
            raise ValueError(
                f"Input layer {input_layer} representing {node_name} is incorrect type"
            )
        if len(next_nodes) == 1:
            # check for appropriate activation function
            type_, maybe_node, maybe_next_nodes = self._nodes[next_nodes[0]]
            if maybe_node.op_type in ["Relu", "Sigmoid", "Tanh", "Softmax", "Softplus"]:
                node = maybe_node
                activation = maybe_node.op_type.lower()
                next_nodes = maybe_next_nodes

        output = create_vars(
            self.scip_model,
            (input_vars.shape[0], weights.shape[-1]),
            vtype="C",
            lb=None,
            ub=None,
            name_prefix=self.unique_naming_prefix
            + f"layer_{self.num_processed_nodes}_{node_name}_",
        )
        self._created_vars.append(output)
        layer = self.add_dense_layer(
            input_vars,
            weights,
            biases,
            activation,
            output,
            unique_naming_prefix=self.unique_naming_prefix
            + f"linear_{self.num_processed_nodes}_{node_name}_",
        )

        # Increment the number of processed nodes and populate the dictionary with the newly created layer
        for node_output in node.output:
            self._node_layer_map[node_output] = layer
        self.num_processed_nodes += 1

        return next_nodes, layer

    def get_error(self, eps=None):
        """
        Returns error in SCIP's solution with respect to the actual output of the ONNX neural network

        Parameters
        ----------
        eps : float or int or None, optional
            The maximum allowed tolerance for a mismatch between the actual predictive model and SCIP.
            If the error is larger than eps an appropriate warning is printed

        Returns
        -------
        error: np.ndarray
            The absolute values of the difference between SCIP's solution and the trained ML model's output given
            the input as defined by SCIP. The matrix is the same dimension as the output of the trained predictor.
            Using ONNX / pyscipopt, the absolute difference between onnxruntime.InferenceSession(input)
            and scip.getVal(output).

        Raises
        ------
        NoSolution
            If SCIP has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution:
            reg = ort.InferenceSession(
                self.predictor.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            # Depending on the ONNX model double or float is expected. Numpy defaults to double.
            t_in = self.input_values
            if reg.get_inputs()[0].type == "tensor(float)":
                t_in = t_in.astype(np.float32)
            # ONNX ModelProto also comes with a batch size. Work around this.
            n_samples_expected = reg.get_inputs()[0].shape[0]
            n_samples = self.input.shape[0]
            if n_samples_expected is None or n_samples_expected == n_samples:
                t_out = reg.run(None, {self.predictor.graph.input[0].name: t_in})
                if isinstance(t_out, list):
                    t_out = t_out[0]
            elif n_samples_expected == 1:
                first_pass = True
                for i in range(n_samples):
                    t_out_temp = reg.run(
                        None, {self.predictor.graph.input[0].name: np.array([t_in[i]])}
                    )
                    if isinstance(t_out_temp, list):
                        t_out_temp = t_out_temp[0]
                    if first_pass:
                        t_out = t_out_temp
                    else:
                        t_out = np.vstack((t_out, t_out_temp))
                    first_pass = False
            else:
                raise ParameterError(
                    f"ONNX GraphProto has input dim {n_samples_expected}. It must be None, 1, or the "
                    f"same as the input {n_samples}"
                )
            # In the case of single large batches the output might all be concatenated. Reshape it.
            if t_out.shape != self.output_values.shape:
                t_out = t_out.reshape(self.output_values.shape)
            if self.output_type == "classification":
                argmax_indices = np.argmax(t_out, axis=1)
                t_out = np.zeros_like(t_out, dtype=int)
                t_out[np.arange(t_out.shape[0]), argmax_indices] = 1
            r_val = np.abs(t_out - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{t_out} != {self.output_values}")
            return r_val
        raise NoSolution()


def _collect_attributes(node):
    """
    Returns the correct values of the attributes stored for the specific node

    Parameters
    ----------
    node :
        The node object from the graph structure of ONNX.


    Returns
    -------
    value_dict : dict
        A dictionary containing the attribute values of the node

    """
    value_dict = {}
    for attr in node.attribute:
        if attr.type == 1:  # FLOAT
            value_dict[attr.name] = attr.f
        elif attr.type == 2:  # INT
            value_dict[attr.name] = int(attr.i)
        elif attr.type == 4:  # TENSOR
            value_dict[attr.name] = numpy_helper.to_array(attr.t)
            pass
        elif attr.type == 7:  # INTS
            value_dict[attr.name] = list(attr.ints)
        else:
            raise RuntimeError(f"unhandled attribute type {attr.type}")
    return value_dict
