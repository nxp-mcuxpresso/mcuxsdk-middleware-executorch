# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain

import numpy as np
from torch.fx import Node
from torch.nn import Parameter

from executorch.backends.nxp.backend.custom_delegation_options import CustomDelegationOptions
from executorch.backends.nxp.backend.edge_helper import node_is_effectively_static_tensor, \
    get_quantization_parameters_for, get_quantization_parameters_for_output_on_index
from executorch.backends.nxp.backend.ir.converter.conversion.common import try_get_input, OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter, Target
from executorch.backends.nxp.backend.ir.converter.tensor_utils import get_input_shape
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.custom_options.pytorch_gru_options import PyTorchGRU, \
    Direction, Activation


class GRUConverter(NodeConverter):
    @staticmethod
    def _is_supported_on_target(
        node: Node,
        target: Target,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions
    ) -> bool:
        match target:
            case Target.RT700:
                # For the `RT700`, the GRU is converted into a custom `PyTorchGRU` operator in the IR.

                if len(node.args[0].meta['val'].shape) != 3:
                    return False  # Unexpected case.

                has_biases, num_layers, dropout, train, bidirectional, batch_first = node.args[3:]
                if batch_first:
                    # If `batch_size == 1`, a `Reshape` operator can be used to turn the GRU into time major mode.
                    batch_size = get_input_shape(node, 0)[0]
                    if batch_size != 1:
                        return False  # Would require a `Transpose` before the input.
                if num_layers != 1:
                    # There is no direct equivalent in the custom TFLite GRU.
                    return False

                input_size = get_input_shape(node, 0)[-1]
                hidden_size = get_input_shape(node, 1)[-1]
                num_layers = node.args[4]
                num_directions = 2 if bidirectional else 1
                if num_layers != 1 and input_size != num_directions * hidden_size:
                    # If there are more layers, the weights for the subsequent layers can have a different shape than
                    #  the first layer. In the custom TFLite GRU for Neutron, the weights must be stacked into a single
                    #  tensor, which is only possible if the shapes match.
                    return False

                if input_size % 8 != 0 or hidden_size % 8 != 0:
                    # The NeutronConverter lowers the `PyTorchGRU` into multiple simpler operations. If the input_size
                    #  or the hidden_size are not a multiple of 8, some "sub-operations" of the `PyTorchGRU` would not
                    #  be delegated.
                    return False

                all_weights = node.args[2]
                weights = []
                stride = 4 if has_biases else 2
                for i in range(0, len(all_weights), stride):
                    weights.append(all_weights[i])
                    weights.append(all_weights[i + 1])

                # Make sure the weights are static, so they can be stacked together.
                if not all(
                    node_is_effectively_static_tensor(w, parameters_mapping)
                    for w in weights
                ):
                    return False

                if has_biases:
                    biases = []
                    for i in range(0, len(all_weights), 4):
                        biases.append(all_weights[i + 2])
                        biases.append(all_weights[i + 3])

                    # Check that the biases have static data, so they can be stacked into a bigger tensor.
                    if not all(
                        node_is_effectively_static_tensor(bias, parameters_mapping)
                        for bias in biases
                    ):
                        return False

                # Check that the quantization parameters of the hidden_state follow the requirements of `PyTorchGRU`.
                h_quant_params = get_quantization_parameters_for(node.args[1])
                y_quant_params = get_quantization_parameters_for_output_on_index(node, 0)
                yh_quant_params = get_quantization_parameters_for_output_on_index(node, 1)
                if any(params is None for params in [h_quant_params, y_quant_params, yh_quant_params]):
                    return False
                if not (h_quant_params == y_quant_params == yh_quant_params):
                    return False

                # All checks passed. The node is supported by the platform.
                return True

            case _:
                return False

    def _get_static_quant_params_from_nodes(self, quant_nodes):
        return tuple(self.context.parameters_mapping[str(node)] for node in quant_nodes)

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions
    ) -> bool:
        # The IR support is strictly platform dependent. The checks are handled in `_is_supported_on_target()`.
        return True

    def _quantization_parameters_valid(self, node: Node):
        # Check that the quantization parameters of the biases follow the requirements of `PyTorchGRU`.
        #  The following code only works for num_layers == 1. The assertion only serves as an indicator to
        #  the developer, to update this code as well in case support for num_layers >= 2 is ever added.

        has_biases, num_layers, _, _, bidirectional, batch_first = node.args[3:]
        all_weights = node.args[2]
        assert num_layers == 1

        weights = []
        stride = 4 if has_biases else 2
        for i in range(0, len(all_weights), stride):
            weights.append(all_weights[i])
            weights.append(all_weights[i + 1])

        x_quant_params = get_quantization_parameters_for(node.args[0])
        h_quant_params = get_quantization_parameters_for(node.args[1])
        w_quant_nodes = get_quantization_parameters_for(node.args[2][0])
        w_quant_params = self._get_static_quant_params_from_nodes(w_quant_nodes)
        r_quant_nodes = get_quantization_parameters_for(node.args[2][1])
        r_quant_params = self._get_static_quant_params_from_nodes(r_quant_nodes)

        if any(params is None for params in [x_quant_params, h_quant_params, w_quant_params, r_quant_params]):
            return False

        if has_biases:
            biases = []
            for i in range(0, len(all_weights), 4):
                biases.append(all_weights[i + 2])
                biases.append(all_weights[i + 3])

            bw_quant_nodes = get_quantization_parameters_for(biases[0])
            bw_quant_params = self._get_static_quant_params_from_nodes(bw_quant_nodes)
            br_quant_nodes = get_quantization_parameters_for(biases[1])
            br_quant_params = self._get_static_quant_params_from_nodes(br_quant_nodes)

            if any(param is None for param in [bw_quant_params, br_quant_params]):
                return False

            if not np.allclose(bw_quant_params[0], x_quant_params[0] * np.array(w_quant_params[0])):
                # w_bias.scale == x.scale * w.scale
                return False

            if not np.allclose(br_quant_params[0], h_quant_params[0] * np.array(r_quant_params[0])):
                # r_bias.scale == h.scale * r.scale
                return False

            if bw_quant_params[1].any().item() or br_quant_params[1].any().item():
                # The zero points must all be 0.
                return False

        if bidirectional:
            # The corresponding forward and reverse weights and biases must have the same quantization.
            assert num_layers == 1
            w_reverse_quant_nodes = get_quantization_parameters_for(weights[2])
            w_reverse_quant_params = self._get_static_quant_params_from_nodes(w_reverse_quant_nodes)
            r_reverse_quant_nodes = get_quantization_parameters_for(weights[3])
            r_reverse_quant_params = self._get_static_quant_params_from_nodes(r_reverse_quant_nodes)
            equal_quantization = [
                np.allclose(w_quant_params[0], w_reverse_quant_params[0]),
                np.allclose(r_quant_params[0], r_reverse_quant_params[0]),
                np.allclose(w_quant_params[1], w_reverse_quant_params[1]),
                np.allclose(r_quant_params[1], r_reverse_quant_params[1]),
            ]

            if not all(equal_quantization):
                return False

            if has_biases:
                assert biases
                assert bw_quant_params
                assert br_quant_params

                bw_reverse_quant_nodes = get_quantization_parameters_for(biases[2])
                bw_reverse_quant_params = self._get_static_quant_params_from_nodes(bw_reverse_quant_nodes)
                br_reverse_quant_nodes = get_quantization_parameters_for(biases[3])
                br_reverse_quant_params = self._get_static_quant_params_from_nodes(br_reverse_quant_nodes)
                equal_quantization = [
                    np.allclose(bw_quant_params[0], bw_reverse_quant_params[0]),
                    np.allclose(br_quant_params[0], br_reverse_quant_params[0]),
                    np.allclose(bw_quant_params[1], bw_reverse_quant_params[1]),
                    np.allclose(br_quant_params[1], br_reverse_quant_params[1]),
                ]
                if not all(equal_quantization):
                    return False

        return True

    def convert(self, node: Node):
        """ Convert 'aten.gru.input' operator to a custom TFLite operator 'PyTorchGRU'.

            The `aten.gru.input` has the following schema:
             aten::gru.input(
                 Tensor input, Tensor hx, Tensor[] params,
                 bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first
             ) -> (Tensor, Tensor)

            The `input` has shape:
                [sequence_length, batch_size, input_size] if `batch_first == False` else
                [batch_size, sequence_length, input_size]

            The `hx` (initial hidden state) has shape:
                [D * num_layers, hidden_size] or [D * num_layers, batch_size, hidden_size]
                 where D = 2 if bidirectional else 1.

            The `params` list contains the weights and biases. The tensors are stored in the following pattern:
            [
                weight_ih_l[k] of shape [3 * hidden_size, input_size] if k==0 else [3 * hidden_size, num_directions * hidden_size],
                weight_hh_l[k] of shape [3 * hidden_size, hidden_size],
                bias_ih_l[k] of shape [3 * hidden_size],
                bias_hh_l[k] of shape [3 * hidden_size]
                ...
            ]
            These 4 elements repeat for every layer (for k in range(num_layers)). If `has_biases` is False, only the
             first 2 elements (the weight_ih_l and weight_hh_l) are present and repeated.
            If the GRU is `bidirectional`, every other set of weights and biases (2 or 4 elements) refers to the reverse
             connections for that layers.


            The custom `PyTorchGRU` has the following inputs:
             - main input (same as `aten.gru.input`).
                -[seq_length, batch_size, input_size], INT8, quantized per tensor, asymmetric, unconstrained scale,
                  unconstrained offset.
             - W (concatenation of `weight_ih_l[k]`, or just the first one if `num_layers` == 1).
                - [num_directions, 3*hidden_size, input_size], INT8, quantized per channel, symmetric (offsets are 0),
                   the quantization axis is axis 1 (with length 3*hidden_size) , unconstrained scales.
             - R (concatenation of `weight_hh_l[k]`, or just the first one if `num_layers` == 1).
                - [num_directions, 3*hidden_size, hidden_size], INT8, quantized per channel, symmetric (offsets are 0),
                   the quantization axis is axis 1 (with length 3*hidden_size) , unconstrained scales.
             - B (concatenation of `bias_ih_l[k]` and `bias_hh_l[k]`).
                - [num_directions, 6*hidden_size], INT32, quantized per channel, symmetric (offsets are 0), the
                   quantization axis is axis 1 (with length 6*hidden_size), constrained scales  X_SCALE * W_SCALES
                   (3 * hidden_sizes scales) concatenated with H_SCALE * R_SCALES (3 * hidden_sizes scales).
             - sequence lengths.
                - [batch_size] (unused).
             - initial_h.
                - [num_directions, batch_size, hidden_size].

            And outputs:
             - Y.
                - [seq_length, num_directions, batch_size, hidden_size].
             - Y_h (same as `aten.gru.input`).
                - [num_directions, batch_size, hidden_size].

            The operands initial_h, Y, Y_h must have the same quantization parameters. INT8, quantized per tensor,
             asymmetric, unconstrained scale, unconstrained offset.
        """
        self.assert_convertible(node)
        quant_params_valid = False
        try:
            quant_params_valid = self._quantization_parameters_valid(node)
        finally:
            if not quant_params_valid:
                raise RuntimeError("GRU node has invalid quantization parameters.")

        t_op = self._create_tflite_op_with_io_tensors(node)
        ops = OpsList(middle_op=t_op)

        # Extract the arguments of the `aten.gru.input`.
        has_biases, num_layers, dropout, train, bidirectional, batch_first = node.args[3:]

        # Extract the inputs of the `aten.gru.input`.
        x = t_op.tmp_inputs[0]
        initial_h = t_op.tmp_inputs[1]

        # Parse the weights.
        # Careful. If support for `num_layers != 1` is added, this code will not work as the inputs are complicated.
        #  The following assertion only serves as an indication to the developers to update this code as well.
        assert num_layers == 1
        w = t_op.tmp_inputs[2]
        r = t_op.tmp_inputs[3]
        if has_biases:
            w_bias = t_op.tmp_inputs[4]
            r_bias = t_op.tmp_inputs[5]
            w_reverse = try_get_input(t_op, 6)
            r_reverse = try_get_input(t_op, 7)
            w_bias_reverse = try_get_input(t_op, 8)
            r_bias_reverse = try_get_input(t_op, 9)
        else:
            w_reverse = try_get_input(t_op, 4)
            r_reverse = try_get_input(t_op, 5)
            w_bias = r_bias = w_bias_reverse = r_bias_reverse = None

        all_biases = [b for b in [w_bias, r_bias, w_bias_reverse, r_bias_reverse] if b is not None]

        # Extract the hyperparameters from the tensor shapes.
        input_size = w.shape[1]
        hidden_size = r.shape[1]
        num_directions = 2 if bidirectional else 1
        batch_size = x.shape[0] if batch_first else x.shape[1]
        sequence_length = x.shape[1] if batch_first else x.shape[0]

        # Prepare the concatenated bias tensor.
        if has_biases:
            bias_data = np.stack(
                [b.tmp_buffer.data for b in all_biases]
            ).reshape([num_directions, 2 * 3 * hidden_size])

            bias = self.builder.create_tensor_for_data(bias_data, 'gru_concatenated_bias_')

            # Compute the quantization parameters of the concatenated bias tensor.
            bias_scales = [b.quantization.scale.vector for b in [w_bias, r_bias]]
            bias_zero_points = [b.quantization.zero_point.vector for b in [w_bias, r_bias]]

            bias_quantization = tflite_model.Quantization(
                scale=tflite_model.Scale(list(chain(*bias_scales))),
                zero_point=tflite_model.ZeroPoint(list(chain(*bias_zero_points))),
                quantized_dimension=1  # The bias is 2D, and quantized per-channel along the second dimension.
            )
            bias.quantization = bias_quantization

        else:
            bias = self.builder.create_null_tensor()  # Empty tensor.

        if bidirectional:
            # Combine the forward and reverse weights together.
            weights_data = np.stack(
                [w_.tmp_buffer.data for w_ in [w, w_reverse]]
            ).reshape([num_directions, 3 * hidden_size, input_size])

            r_weights_data = np.stack(
                [r_.tmp_buffer.data for r_ in [r, r_reverse]]
            ).reshape([num_directions, 3 * hidden_size, hidden_size])

            w.tmp_buffer.data = weights_data
            r.tmp_buffer.data = r_weights_data

        # Reshape the weight tensors to match the shapes expected by the custom TFLite GRU.
        new_w_shape = [num_directions, 3 * hidden_size, input_size]
        w.shape.vector = new_w_shape
        w.tmp_buffer.data.reshape(new_w_shape)

        new_r_shape = [num_directions, 3 * hidden_size, hidden_size]
        r.shape.vector = new_r_shape
        r.tmp_buffer.data.reshape(new_r_shape)

        # Make sure the quantized dimension matches the Neutron requirement.
        w.quantization.quantized_dimension = 1
        r.quantization.quantized_dimension = 1

        # Set the inputs in the order required by the custom PyTorchGRU.
        sequence_lengths = self.builder.create_null_tensor()
        t_op.tmp_inputs = [
            x,
            w,
            r,
            bias,
            sequence_lengths,
            initial_h
        ]

        if batch_first:
            assert batch_size == 1  # This was checked before. The `assert` alerts us in case the code is modified.
            # Add a `Reshape` to make the operator time major.
            time_major_input_shape = [sequence_length, batch_size, input_size]
            ops.add_pre(self.builder.create_reshape_before(t_op, 0, time_major_input_shape))

        # Assign the custom TFLite options.
        t_op.custom_options = PyTorchGRU(
            hidden_size,
            direction=Direction.bidirectional if bidirectional else Direction.forward,
            activations=(
                Activation.sigmoid, Activation.tanh, Activation.sigmoid, Activation.tanh
            ) if bidirectional else (
                Activation.sigmoid, Activation.tanh
            )
        )

        y = t_op.tmp_outputs[0]
        if y.rank != 4:
            # The custom TFLite GRU creates the main output with shape
            #  [sequence_length, num_directions, batch_size, hidden_size],
            # but the `aten.gru.input` can produce an output with multiple different shapes, based on the parameters.
            # Insert a `Reshape` after, to make sure the final output matches the `aten.gru.input`.

            # This implementation only works when `batch_first == False` or when `batch_size == 1`, which is assured
            #  above, but check it here in case the converter is updated.
            assert (not batch_first) or (batch_size == 1)

            custom_gru_output_shape = [sequence_length, num_directions, batch_size, hidden_size]
            aten_gru_output_shape = y.shape.vector.copy()

            y.shape.vector = custom_gru_output_shape
            ops.add_post(
                self.builder.create_reshape_after(t_op, 0, aten_gru_output_shape)
            )

        self.builder.append_operators(ops.flatten())
