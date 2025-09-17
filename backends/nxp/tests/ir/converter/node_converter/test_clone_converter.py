# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from torch import nn
from torch.export import ExportedProgram

from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import PermuteCopyConverter
from executorch.backends.nxp.backend.ir.edge_passes.remove_io_quant_ops_pass import RemoveIOQuantOpsPass
from executorch.backends.nxp.edge_passes.move_auxiliary_operator_into_separate_qdq_cluster_pass import \
    MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass
from executorch.backends.nxp.edge_passes.nxp_edge_pass_manager import NXPEdgePassManager
from executorch.backends.nxp.neutron_partitioner import QDQClusterRecognizer
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.tests import executors
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program, \
    get_random_calibration_inputs, to_model_input_spec, _quantize_model
from executorch.backends.nxp.tests.executors import convert_run_compare, graph_contains_any_of_ops, \
    ToChannelLastPreprocess, \
    ToChannelFirstPreprocess, OverrideTargetSupportCheck
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.program._program import to_edge_with_preserved_ops
from executorch.extension.export_util.utils import _to_core_aten


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class SingleConvBlockWithDropout(torch.nn.Module):
    def __init__(self, conv_in_channels: int = 3, perform_inplace_dropout: bool = False):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=conv_in_channels, out_channels=64, kernel_size=(4, 4)),
            torch.nn.ReLU(),
            torch.nn.Dropout(inplace=perform_inplace_dropout)
        )

    def forward(self, x):
        return self.block(x)


class KWSFinalBlock(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        pool_size = (25, 5)
        self.block = torch.nn.Sequential(
            self.conv_sep_dw(inp=input_shape[1], oup=64),
            nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=10)
        )

    def conv_sep_dw(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=3, padding=1, groups=inp),
            nn.BatchNorm2d(num_features=inp, eps=1e-3, momentum=0.01),
            nn.ReLU(),

            nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=oup, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class TransposeReshapeModel(nn.Module):

    def __init__(self, new_shape: list[int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        # `x` should be 4D.

        x = torch.add(x, x)
        x = torch.transpose(x, 1, 2)
        # A `clone(memory_format=contiguous)` will be added here during the lowering to edge dialect.
        x = torch.reshape(x, self.new_shape)

        return x


class PermuteCopyReshapeModel(nn.Module):

    def __init__(self, new_shape: list[int], permutation: list[int]):
        super().__init__()
        self.new_shape = new_shape
        self.permutation = permutation

    def forward(self, x):
        # `x` should be 4D.

        x = torch.add(x, x)
        x = torch.permute(x, self.permutation)
        # A `clone(memory_format=contiguous)` will be added here during the lowering to edge dialect.
        x = torch.reshape(x, self.new_shape)
        x = torch.add(x, x)

        return x


@pytest.mark.parametrize('inplace_dropout', [False, True])
@pytest.mark.parametrize('input_shape', [(1, 3, 128, 128), (1, 3, 256, 256)])
def test_conv_dropout_quant(mocker, inplace_dropout: bool, input_shape: tuple[int]):
    model = SingleConvBlockWithDropout(conv_in_channels=input_shape[1], perform_inplace_dropout=inplace_dropout).eval()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(model, input_shape).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    assert not graph_contains_any_of_ops(graph=quantized_program.graph, ops=[exir_ops.edge.aten.clone.default])

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(exported_program,
                        tfl_model=tflite_flatbuffers_model,
                        tflite_input_preprocess=ToChannelLastPreprocess(),
                        tflite_output_preprocess=ToChannelFirstPreprocess(),
                        input_data=input_data,
                        atol=1.)


@pytest.mark.parametrize('inplace_dropout', [False, True])
def test_clone_pool_view_copy_quant(mocker, inplace_dropout: bool, input_shape: tuple[int] = (1, 64, 25, 5)):
    model = KWSFinalBlock(input_shape).eval()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    quantized_program = to_quantized_edge_program(model, input_shape).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    assert not graph_contains_any_of_ops(graph=quantized_program.graph, ops=[exir_ops.edge.aten.clone.default])

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(exported_program,
                        tfl_model=tflite_flatbuffers_model,
                        tflite_input_preprocess=ToChannelLastPreprocess(),
                        input_data=input_data,
                        atol=1.)


def test_clone__to_contiguous_format():
    input_shape = (2, 4, 6, 8)  # Will be permuted to [2, 6, 4, 8]
    new_shape = [3, 4, 16, 2]

    model = TransposeReshapeModel(new_shape).eval()

    calibration_inputs = get_random_calibration_inputs(to_model_input_spec(input_shape))
    example_input = calibration_inputs[0]

    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)
    exir_program_aten_quant = _quantize_model(exir_program_aten, NeutronQuantizer(), calibration_inputs)
    edge_program_manager = to_edge_with_preserved_ops(_to_core_aten(exir_program_aten_quant, example_input))

    # Make sure the `aten.clone` was inserted as expected.
    nodes = list(edge_program_manager.exported_program().graph.nodes)
    assert 'clone' in nodes[9].name
    assert nodes[9].kwargs['memory_format'] == torch.contiguous_format

    # Move the `clone` out of the cluster with the `view_copy`.
    edge_program_manager = NXPEdgePassManager(edge_program_manager, [
        MoveLeadingAuxiliaryOperatorIntoSeparateQDQClusterPass
    ]).transform()

    # Tag QDQ clusters, so the conversion works correctly.
    QDQClusterRecognizer().tag_qdq_clusters(list(edge_program_manager.exported_program().graph.nodes))
    edge_program_manager.exported_program().graph_module.recompile()
    edge_program_manager = edge_program_manager.transform(
        [RemoveIOQuantOpsPass(edge_program_manager=edge_program_manager)]
    )

    # Convert to the IR.
    converted_model, _ = EdgeProgramToIRConverter().convert_program(edge_program_manager.exported_program())

    # Make sure the IR version produces the same outputs.
    executors.convert_run_compare(
        edge_program_manager.exported_program(),
        np.random.random_integers(0, 255, input_shape).astype('int8'),
        tfl_model=converted_model
    )


def test_clone__to_contiguous_format__non_delegated_permute_copy():
    input_shape = (2, 4, 6, 8)
    new_shape = [3, 4, 16, 2]
    permutation = [3, 2, 1, 0]  # Unsupported by default.

    model = PermuteCopyReshapeModel(new_shape, permutation).eval()

    # Prohibit `permute_copy` delegation in case support for the permutation is added in the future.
    with OverrideTargetSupportCheck(PermuteCopyConverter, new_target_support_check=lambda *_: False):
        ep = to_quantized_edge_program(model, input_shape).exported_program()

    nodes = list(ep.graph.nodes)
    assert not graph_contains_any_of_ops(ep.graph, [exir_ops.edge.aten.clone.default])
    assert nodes[4].name == 'executorch_call_delegate_1'
    assert nodes[7].target == exir_ops.edge.aten.permute_copy.default
    assert nodes[9].name == 'executorch_call_delegate'
