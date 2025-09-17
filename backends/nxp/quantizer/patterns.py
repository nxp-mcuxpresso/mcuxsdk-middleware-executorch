# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import fx
from torch._ops import OpOverload
from torch.ao.quantization import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    SharedQuantizationSpec,
    QuantizationSpec,
)

from executorch.backends.nxp.quantizer.utils import get_bias_qparams


@dataclass
class NodeArgsIdx:
    """
    Specifies indexes to args paramater of Node in node input annotation.


    Attributes:
        idx (int): Index to Node's args paramater (list). Selects an input Node or a list of Nodes at the index.
        inner_idx (int): If specified, index to a list pointed by 'idx' attribute. Selects an input Node at the index.
                         Default: None.
    """
    idx: int
    inner_idx: int = None


@dataclass
class PartitionAnchors:
    """
    All fields except output are lists of (node, node_args_idx) or (node, node_args_idx, quantization_spec) tuples,
    where node is from the given partition and node.args[node_args_idx] is an input to the partition. Assumes
    a single output.

    Quantizer uses inputs, weights and biases for quantization annotation. The others
    field contains tensor inputs that aren't quantized, and the literals fields contains
    is used for other types of input values as well as handling default parameters.
    """

    # Inputs can share quantization parameters
    inputs: list[
        tuple[fx.Node, NodeArgsIdx] |
        tuple[fx.Node, NodeArgsIdx, SharedQuantizationSpec],
    ] = field(default_factory=list)
    weights: list[
        tuple[fx.Node, NodeArgsIdx] |
        tuple[fx.Node, NodeArgsIdx, QuantizationSpec],
    ] = field(default_factory=list)
    biases: list[
        tuple[fx.Node, NodeArgsIdx] |
        tuple[fx.Node, NodeArgsIdx, DerivedQuantizationSpec],
    ] = field(default_factory=list)
    others: list[tuple[fx.Node, NodeArgsIdx]] = field(default_factory=list)
    literals: list[tuple[fx.Node, NodeArgsIdx]] = field(default_factory=list)
    output: list[
        tuple[fx.Node] |
        tuple[fx.Node, FixedQParamsQuantizationSpec | SharedQuantizationSpec],
    ] = field(default_factory=list)
    empty: bool = False


class QuantizationPattern(ABC):
    @abstractmethod
    def partition_types(self) -> list[OpOverload]:
        """
        list of types to be passed to find_sequential_partitions_aten.
        """
        pass

    @abstractmethod
    def get_anchors(
        self, gm: torch.fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        pass


class SharedSpecPattern(QuantizationPattern):
    """
    Quantization pattern for shared quantization.

    The quantization is derived from the previous node quantization and the input and output shares the same
    quantization parameters (scale and zero-point).
    """

    @abstractmethod
    def partition_types(self) -> list[OpOverload]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]
        assert len(fused_partition[0].input_nodes) == 1
        prev_node = fused_partition[0].input_nodes[0]

        # Previous node was not quantized => we are not able to share q-params
        if "quantization_annotation" not in prev_node.meta:
            return None

        qspec = SharedQuantizationSpec(prev_node)

        return PartitionAnchors(
            inputs=[(node, NodeArgsIdx(0))],
            weights=[],
            biases=[],
            output=[
                (node, qspec),
            ],
        )


def get_anchors_for_fixed_quant_specs(
    fused_partition: list[fx.GraphModule],
    scale: float,
    zero_point: int,
    quant_min: int = -128,
    quant_max: int = 127,
) -> PartitionAnchors:
    node = fused_partition[0].nodes[-1]
    assert len(fused_partition[0].input_nodes) == 1

    qspec = FixedQParamsQuantizationSpec(
        dtype=torch.int8,
        scale=scale,
        zero_point=zero_point,
        quant_min=quant_min,
        quant_max=quant_max,
        qscheme=torch.per_tensor_affine,
    )

    return PartitionAnchors(
        inputs=[(node, NodeArgsIdx(0))],
        weights=[],
        biases=[],
        output=[(node, qspec), ],
    )


class CatPattern(QuantizationPattern):
    """
    Quantizer for the Cat operator. The pattern is designed for the `NeutronAtenQuantizer`.

    The node can have an arbitrary number of inputs, which are all quantized.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.cat.default]

    def get_anchors(self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]

        quantized_input = None
        for prev_node in node.args[0]:
            if "quantization_annotation" in prev_node.meta:
                quantized_input = prev_node
                break

        if quantized_input is not None:
            inputs = []
            for idx, input_ in enumerate(node.args[0]):
                inputs.append((node, NodeArgsIdx(0, idx), SharedQuantizationSpec(quantized_input)))
            outputs = [(node, SharedQuantizationSpec(quantized_input))]

        else:
            # No previous node was quantized => we are not able to share q-params. The conversion to IR will have to
            #  re-quantize the inputs if necessary.
            inputs = [(node, NodeArgsIdx(0, idx)) for idx in range(len(node.args[0]))]
            outputs = [(node,)]

        return PartitionAnchors(
            inputs=inputs,
            weights=[],
            biases=[],
            output=outputs,
        )


class AddTensorPattern(QuantizationPattern):
    """
    Quantization pattern for Add Tensor quantization. Accepts 1 or 2 input nodes.

    Basic quantization for all inputs and output.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.add.Tensor]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]
        inputs = [(node, NodeArgsIdx(0))]
        if len(fused_partition[0].input_nodes) == 2:
            inputs = [(node, NodeArgsIdx(0)), (node, NodeArgsIdx(1))]

        return PartitionAnchors(
            inputs=inputs,
            weights=[],
            biases=[],
            output=[(node,)],
        )


class AddmmPattern(QuantizationPattern):
    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.addmm.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        addmm_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (addmm_node.args[1], addmm_node),
                (addmm_node.args[2], addmm_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2 ** 31),
            quant_max=2 ** 31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        return PartitionAnchors(
            inputs=[(addmm_node, NodeArgsIdx(1))],
            weights=[(addmm_node, NodeArgsIdx(2))],
            biases=[(addmm_node, NodeArgsIdx(0), bias_qspec)],
            output=[(addmm_node,)],
        )


class AvgPoolPattern(SharedSpecPattern):
    """
    Quantizer for AvgPool2D operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.avg_pool2d.default]


class ConvPattern(QuantizationPattern):
    @abstractmethod
    def partition_types(self) -> list[OpOverload]:
        pass

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        conv_node = fused_partition[0].nodes[-1]

        bias_quantization_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv_node.args[0], conv_node),
                (conv_node.args[1], conv_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2 ** 31) + 1,
            quant_max=2 ** 31 - 1,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr,
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv_node.args) > 2 and conv_node.args[2] is not None:
            bias = [(conv_node, NodeArgsIdx(2), bias_quantization_qspec)]

        return PartitionAnchors(
            inputs=[(conv_node, NodeArgsIdx(0))],
            weights=[(conv_node, NodeArgsIdx(1), weight_quantization_spec)],
            biases=bias,
            output=[(conv_node,)],
        )


class Conv1dPattern(ConvPattern):
    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv1d.default]


class ConvTranspose1dPattern(ConvPattern):
    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv_transpose1d.default]


class Conv2dPattern(ConvPattern):
    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv2d.default]


class ConvTranspose2dPattern(QuantizationPattern):
    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv_transpose2d.input]

    def get_anchors(
            self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        conv_node = fused_partition[0].nodes[-1]

        bias_quantization_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv_node.args[0], conv_node),
                (conv_node.args[1], conv_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2 ** 31) + 1,
            quant_max=2 ** 31 - 1,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr,
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=1,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(conv_node.args) > 2 and conv_node.args[2] is not None:
            bias = [(conv_node, NodeArgsIdx(2), bias_quantization_qspec)]

        return PartitionAnchors(
            inputs=[(conv_node, NodeArgsIdx(0))],
            weights=[(conv_node, NodeArgsIdx(1), weight_quantization_spec)],
            biases=bias,
            output=[(conv_node,)],
        )


class TanhPattern(QuantizationPattern):
    """
    Quantizer for Tanh operator.

    The quantization of Tanh output is fixed to scale 1/128, zero point 0, dtype int8.
    """

    def partition_types(self):
        return [torch.ops.aten.tanh.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        return get_anchors_for_fixed_quant_specs(fused_partition, scale=1.0 / 128.0, zero_point=0)


class TanhInPlacePattern(QuantizationPattern):
    """
    Quantizer for inplace version of Tanh operator (torch.tanh_).

    The quantization of Tanh output is fixed to scale 1/128, zero point 0, dtype int8.
    """

    def partition_types(self):
        return [torch.ops.aten.tanh_.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        return get_anchors_for_fixed_quant_specs(fused_partition, scale=1.0 / 128.0, zero_point=0)


class HardTanhPattern(QuantizationPattern):
    """ Quantizer for HardTanh operator. """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.hardtanh.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]

        return PartitionAnchors(
            inputs=[(node, NodeArgsIdx(0))],
            weights=[],
            biases=[],
            output=[(node,)],
        )


class HardTanhInPlacePattern(QuantizationPattern):
    """ Quantizer for HardTanh operator with param inplace=True.  """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.hardtanh_.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors | None:
        node = fused_partition[0].nodes[-1]

        return PartitionAnchors(
            inputs=[(node, NodeArgsIdx(0))],
            weights=[],
            biases=[],
            output=[(node,)],
        )


class ViewPattern(SharedSpecPattern):
    """
    Quantizer for View operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.view.default]


class FlattenPattern(SharedSpecPattern):
    """
    Quantizer for Flatten operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.flatten.using_ints]


class AdaptiveAvgPoolPattern(SharedSpecPattern):
    """
    Quantizer for AdaptiveAvgPool2D operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.adaptive_avg_pool2d.default]


class AbsPattern(SharedSpecPattern):
    """
    Quantizer for Abs operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.abs.default]


class MeanDimPattern(SharedSpecPattern):
    """
    Quantizer for Mean Dim operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.mean.dim]


class DropoutPattern(SharedSpecPattern):
    """
    Quantizer for Dropout operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.dropout.default]


class LinearPattern(QuantizationPattern):
    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.linear.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__ge...
        linear_node = fused_partition[0].nodes[-1]

        bias_qspec = DerivedQuantizationSpec(
            derived_from=[
                (linear_node.args[0], linear_node),
                (linear_node.args[1], linear_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2 ** 31),
            quant_max=2 ** 31 - 1,
            qscheme=torch.per_tensor_affine,
        )

        # Keep bias empty if not supplied
        bias = []
        if len(linear_node.args) > 2:
            bias = [(linear_node, NodeArgsIdx(2), bias_qspec)]

        return PartitionAnchors(
            inputs=[(linear_node, NodeArgsIdx(0))],
            weights=[(linear_node, NodeArgsIdx(1))],
            # pyre-fixme[6]: Incompatible parameter type
            biases=bias,
            output=[(linear_node,)],
        )


class MaxPoolPattern(SharedSpecPattern):
    """
    Quantizer for MaxPool2D operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.max_pool2d.default]


class PadPattern(SharedSpecPattern):
    """
    Quantizer for Pad operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.pad.default]


class PermutePattern(SharedSpecPattern):
    """
    Quantizer for Permute operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.permute.default]


class TransposeIntPattern(SharedSpecPattern):
    """
    Quantizer for Transpose Int operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.transpose.int]


class ReluPattern(SharedSpecPattern):
    """
    Quantizer for Relu operator. Shared quantization spec is selected, as ReLU usually follows computation layer.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.relu.default]


class ReluInPlacePattern(SharedSpecPattern):
    """
    Quantizer for Relu operator with param inplace=True. Shared quantization spec is selected, as ReLU usually
    follows computation layer.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.relu_.default]


class ReshapePattern(SharedSpecPattern):
    """
    Quantizer for Reshape operator.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.reshape.default]


class SoftMaxPattern(QuantizationPattern):
    """
    Quantizer for Softmax operator.

    The quantization of Softmax output is fixed to scale 1/256, zero point -128, dtype int8.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.softmax.int]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        return get_anchors_for_fixed_quant_specs(fused_partition, scale=1.0 / 256.0, zero_point=-128)


class SigmoidPattern(QuantizationPattern):
    """
    Quantizer for Sigmoid operator.

    The quantization of Sigmoid output is fixed to scale 1/256, zero point -128, dtype int8.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.sigmoid.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        return get_anchors_for_fixed_quant_specs(fused_partition, scale=1.0 / 256.0, zero_point=-128)


class GruInputPattern(QuantizationPattern):
    """
    Quantization pattern for Gru Input quantization. Accepts 2 input nodes.

    Basic quantization for all inputs and outputs.
    """

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.gru.input]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        gru_node = fused_partition[0].nodes[-1]
        hidden_state = gru_node.args[1]
        has_biases, num_layers, dropout, train, bidirectional, batch_first = gru_node.args[3:]

        if num_layers != 1:
            # Quantization implementation should be ready for num_layers > 1, but we don't
            # have quantization specification for this case yet.
            raise NotImplementedError("Quantization of GRU with num_layers != 1 is currently not supported.")

        inputs = [(gru_node, NodeArgsIdx(0)), (gru_node, NodeArgsIdx(1))]

        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        # [*] = optional
        # w_* = weight
        # b_* = bias
        # args[2] = [w_ih, w_hh, [b_ih, b_hh], [w_ih_reverse, w_hh_reverse, [b_ih_reverse, b_hh_reverse] ], ...]
        layer_offset = 2
        layer_offset = layer_offset * 2 if bidirectional else layer_offset
        layer_offset = layer_offset * 2 if has_biases else layer_offset

        weights = []
        biases = []

        for idx in range(num_layers):
            base_idx = idx * layer_offset

            weight_ih = gru_node.args[2][base_idx]
            weight_hh = gru_node.args[2][base_idx + 1]

            weights.append((gru_node, NodeArgsIdx(2, base_idx + 0), weight_quantization_spec))  # weight_ih
            weights.append((gru_node, NodeArgsIdx(2, base_idx + 1), weight_quantization_spec))  # weight_hh

            if has_biases and bidirectional:
                # weight_ih_reverse
                weight_ih_spec = SharedQuantizationSpec((weight_ih, gru_node))
                weights.append((gru_node, NodeArgsIdx(2, base_idx + 4), weight_ih_spec))

                # weight_hh_reverse
                weight_hh_spec = SharedQuantizationSpec((weight_hh, gru_node))
                weights.append((gru_node, NodeArgsIdx(2, base_idx + 5), weight_hh_spec))
            elif bidirectional:
                # weight_ih_reverse
                weight_ih_spec = SharedQuantizationSpec((weight_ih, gru_node))
                weights.append((gru_node, NodeArgsIdx(2, base_idx + 2), weight_ih_spec))

                # weight_hh_reverse
                weight_hh_spec = SharedQuantizationSpec((weight_hh, gru_node))
                weights.append((gru_node, NodeArgsIdx(2, base_idx + 3), weight_hh_spec))

            if has_biases:
                bias_ih = gru_node.args[2][base_idx + 2]
                bias_hh = gru_node.args[2][base_idx + 3]

                bias_qspec_ih = DerivedQuantizationSpec(
                    derived_from=[
                        (gru_node.args[0], gru_node),
                        (weight_ih, gru_node)
                    ],
                    derive_qparams_fn=get_bias_qparams,
                    dtype=torch.int32,
                    quant_min=-(2 ** 31) + 1,
                    quant_max=2 ** 31 - 1,
                    qscheme=torch.per_channel_symmetric,
                    ch_axis=0,
                )

                bias_qspec_hh = DerivedQuantizationSpec(
                    derived_from=[
                        (hidden_state, gru_node),
                        (weight_hh, gru_node)
                    ],
                    derive_qparams_fn=get_bias_qparams,
                    dtype=torch.int32,
                    quant_min=-(2 ** 31) + 1,
                    quant_max=2 ** 31 - 1,
                    qscheme=torch.per_channel_symmetric,
                    ch_axis=0,
                )

                biases.append((gru_node, NodeArgsIdx(2, base_idx + 2), bias_qspec_ih))  # bias_ih
                biases.append((gru_node, NodeArgsIdx(2, base_idx + 3), bias_qspec_hh))  # bias_hh

                if bidirectional:
                    # bias_ih_reverse
                    bias_ih_spec = SharedQuantizationSpec((bias_ih, gru_node))
                    biases.append((gru_node, NodeArgsIdx(2, base_idx + 6), bias_ih_spec))

                    # bias_hh_reverse
                    bias_hh_spec = SharedQuantizationSpec((bias_hh, gru_node))
                    biases.append((gru_node, NodeArgsIdx(2, base_idx + 7), bias_hh_spec))

        # Hidden state input, hidden state output and output must share same quantization params
        output_qspec = SharedQuantizationSpec((hidden_state, gru_node))
        # GRU has 2 outputs, so annotate getitem nodes outputs instead of GRU node itself
        outputs = [(user, output_qspec) for user in gru_node.users]

        return PartitionAnchors(
            inputs=inputs,
            weights=weights,
            biases=biases,
            output=outputs,
        )
