# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import kgb
import numpy as np
import torch
import unittest

from torch import fx
from torch._ops import OpOverload
from torch.ao.quantization import PerChannelMinMaxObserver, MinMaxObserver
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec, QuantizationSpec,
)
from torch.export import ExportedProgram

from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronAtenQuantizer, wgt_qspec, act_qspec
from executorch.backends.nxp.quantizer.patterns import QuantizationPattern, PartitionAnchors, NodeArgsIdx
from executorch.backends.nxp.quantizer.utils import get_bias_qparams
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import convert_run_compare, ToChannelFirstPreprocess, \
    ToChannelLastPreprocess
from executorch.backends.nxp.tests.models import Conv2dModule
from executorch.backends.nxp.tests.test_quantizer import _get_target_name


class Conv2dPatternPerChannel(QuantizationPattern):

    def __init__(self, is_per_channel: bool):
        super().__init__()
        self.is_per_channel = is_per_channel

    def partition_types(self) -> list[OpOverload]:
        return [torch.ops.aten.conv2d.default]

    def get_anchors(
        self, gm: fx.GraphModule, fused_partition: list[fx.GraphModule]
    ) -> PartitionAnchors:
        conv2d_node = fused_partition[0].nodes[-1]

        bias_qscheme = torch.per_channel_symmetric if self.is_per_channel else torch.per_tensor_symmetric
        bias_quantization_qspec = DerivedQuantizationSpec(
            derived_from=[
                (conv2d_node.args[0], conv2d_node),
                (conv2d_node.args[1], conv2d_node),
            ],
            derive_qparams_fn=get_bias_qparams,
            dtype=torch.int32,
            quant_min=-(2 ** 31) + 1,
            quant_max=2 ** 31 - 1,
            qscheme=bias_qscheme,
            ch_axis=0,
        )

        weight_qscheme = torch.per_channel_symmetric if self.is_per_channel else torch.per_tensor_symmetric
        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver if self.is_per_channel else MinMaxObserver
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr,
            quant_min=-127,
            quant_max=127,
            qscheme=weight_qscheme,
            ch_axis=0,
        )

        return PartitionAnchors(
            inputs=[(conv2d_node, NodeArgsIdx(0))],
            weights=[(conv2d_node, NodeArgsIdx(1), weight_quantization_spec)],
            biases=[(conv2d_node, NodeArgsIdx(2), bias_quantization_qspec)],
            output=[(conv2d_node,)],
        )


class GruModule(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False
    ):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional
        )

    def forward(self, input_, h_0):
        # `input_` has shape
        #   [sequence_length, batch_size] or [sequence_length, batch_size, input_size]  if `batch_first` is False
        #   [batch_size, sequence_length, input_size]  if `batch_first` is True
        # `h_0` has shape [D * num_layers, hidden_size] or [D * num_layers, batch_size, hidden_size]
        #   where `D` is equal to 2 if bidirectional==True otherwise 1
        return self.gru(input_, h_0)


class TestPerChannelConversion(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests

    def test_per_channel_convolution(self):
        with kgb.spy_on(EdgeProgramToIRConverter.convert_program, call_original=True) as converter_spy:
            model = Conv2dModule(in_channels=8, out_channels=32, kernel_size=5, padding=3)
            input_shape = (1, 8, 32, 32)

            static_qconfig = QuantizationConfig(act_qspec, act_qspec, wgt_qspec, None)
            quantizer = lambda: NeutronAtenQuantizer(Conv2dPatternPerChannel(is_per_channel=True), static_qconfig)
            _ = to_quantized_edge_program(model, input_shape, get_quantizer_fn=quantizer)

            tflite_flatbuffers_model, io_formats = converter_spy.calls[-1].return_value
            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

            convert_run_compare(exported_program, tflite_input_preprocess=ToChannelLastPreprocess(),
                                tfl_model=tflite_flatbuffers_model,
                                tflite_output_preprocess=ToChannelFirstPreprocess(), input_data=input_data, atol=1.)

            nodes = list(exported_program.graph.nodes)

            assert _get_target_name(nodes[8]).endswith('quantized_decomposed.dequantize_per_channel.default')
            assert _get_target_name(nodes[9]).endswith('quantized_decomposed.dequantize_per_channel.default')
            assert nodes[10].name == 'aten_convolution_default'

    def test_per_channel_gru(self):
        with kgb.spy_on(EdgeProgramToIRConverter.convert_program, call_original=True) as converter_spy:
            input_size = 768
            hidden_size = 128
            sequence_length = 32
            batch_size = 1
            num_layers = 1  # Other values are not supported.
            D = 1  # One directional.
            input_shapes = [
                (sequence_length, batch_size, input_size),
                (D * num_layers, batch_size, hidden_size)
            ]
            model = GruModule(input_size, hidden_size=hidden_size, bidirectional=False, bias=True, num_layers=num_layers).eval()

            _ = to_quantized_edge_program(model, input_shapes).exported_program()

            exported_program: ExportedProgram = converter_spy.calls[-1].args[0]

            # Make sure the GRU was per-channel quantized
            nodes = list(exported_program.graph.nodes)
            assert nodes[15].target.__name__ == 'quantized_decomposed.dequantize_per_tensor.default'
            assert nodes[16].target.__name__ == 'quantized_decomposed.dequantize_per_channel.default'
            assert nodes[17].target.__name__ == 'quantized_decomposed.dequantize_per_channel.default'
            assert nodes[18].target.__name__ == 'quantized_decomposed.dequantize_per_channel.default'
            assert nodes[19].target.__name__ == 'quantized_decomposed.dequantize_per_channel.default'
            assert nodes[20].target.__name__ == 'aten.gru.input'
            assert nodes[23].target.__name__ == 'quantized_decomposed.quantize_per_tensor.default'


    @classmethod
    def setUpClass(cls):
        torch.manual_seed(25)
        np.random.seed(25)
