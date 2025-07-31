# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from torch.export import ExportedProgram

from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass_manager import NXPPyTorchPassManager
from executorch.backends.nxp.pytorch_passes.split_gru_based_on_num_layers import SplitGRUBasedOnNumLayers
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program, to_model_input_spec
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


def get_gru_input_shapes(
    input_size=8,
    hidden_size=8,
    sequence_length=8,
    batch_size=8,
    num_layers=1,
    D=1,  # Unidirectional.
):
    input_shapes = [
        (batch_size, sequence_length, input_size),
        (D * num_layers, batch_size, hidden_size)
    ]

    return input_shapes, input_size, hidden_size, sequence_length


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


@pytest.mark.parametrize('num_layers', [2, 3, 8], ids=lambda num_layers: f'num_layers = {num_layers}')
def test_gru_splitting__with_bias(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(num_layers=num_layers)
    model = GruModule(input_size, hidden_size, num_layers=num_layers).eval()

    example_input = tuple(torch.ones(input_shape) for input_shape in input_shapes)
    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)

    pre_pass_output = [t.detach() for t in exir_program_aten(*example_input)]
    assert len(exir_program_aten.graph.nodes) == 6 + (num_layers) * 4
    assert len([n for n in exir_program_aten.graph.nodes if 'gru' in n.name]) == 1  # Just 1 `GRU` in the model.

    # Run pre-processing passes of the float32 aten dialect program.
    pytorch_pass_manager = NXPPyTorchPassManager(exir_program_aten, [
        SplitGRUBasedOnNumLayers
    ])
    pytorch_pass_manager.run()

    post_pass_output = [t.detach() for t in exir_program_aten(*example_input)]
    assert len(exir_program_aten.graph.nodes) == 5 + (num_layers) * 8
    assert len([n for n in exir_program_aten.graph.nodes if 'gru' in n.name]) == num_layers  # Many `GRU` nodes.

    assert np.allclose(pre_pass_output[0], post_pass_output[0]), 'Main outputs differ'
    assert np.allclose(pre_pass_output[1], post_pass_output[1]), 'Hidden outputs differ'


@pytest.mark.parametrize('num_layers', [2, 3], ids=lambda num_layers: f'num_layers = {num_layers}')
def test_gru_splitting__with_bias__delegation(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(num_layers=num_layers)
    model = GruModule(input_size, hidden_size, num_layers=num_layers).eval()

    edge_program = to_quantized_edge_program(model, to_model_input_spec(input_shapes)).exported_program()

    nodes = list(edge_program.graph.nodes)
    assert not graph_contains_any_of_ops(edge_program.graph, [torch.ops.aten.gru.input])
    delegate_call_index = num_layers * 2 + 7
    assert nodes[delegate_call_index].name == 'executorch_call_delegate'
    # The delegate call produces every GRU's `Y_h` output, and last GRU's `Y` output (num_layers + 1 in total).
    assert len(nodes[delegate_call_index].users.keys()) == num_layers + 1


@pytest.mark.parametrize('num_layers', [2, 5], ids=lambda num_layers: f'num_layers = {num_layers}')
def test_gru_splitting__no_bias(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(num_layers=num_layers)
    model = GruModule(input_size, hidden_size, num_layers=num_layers, bias=False).eval()

    example_input = tuple(torch.ones(input_shape) for input_shape in input_shapes)
    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)

    pre_pass_output = [t.detach() for t in exir_program_aten(*example_input)]
    assert len(exir_program_aten.graph.nodes) == 6 + (num_layers) * 2
    assert len([n for n in exir_program_aten.graph.nodes if 'gru' in n.name]) == 1  # Just 1 `GRU` in the model.

    # Run pre-processing passes of the float32 aten dialect program.
    pytorch_pass_manager = NXPPyTorchPassManager(exir_program_aten, [
        SplitGRUBasedOnNumLayers
    ])
    pytorch_pass_manager.run()

    post_pass_output = [t.detach() for t in exir_program_aten(*example_input)]
    assert len(exir_program_aten.graph.nodes) == 5 + (num_layers) * 6
    assert len([n for n in exir_program_aten.graph.nodes if 'gru' in n.name]) == num_layers  # Many `GRU` nodes.

    assert np.allclose(pre_pass_output[0], post_pass_output[0]), 'Main outputs differ'
    assert np.allclose(pre_pass_output[1], post_pass_output[1]), 'Hidden outputs differ'


@pytest.mark.parametrize('num_layers', [2, 3], ids=lambda num_layers: f'num_layers = {num_layers}')
def test_gru_splitting__no_bias__delegation(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(num_layers=num_layers)
    model = GruModule(input_size, hidden_size, num_layers=num_layers, bias=False).eval()

    edge_program = to_quantized_edge_program(model, to_model_input_spec(input_shapes)).exported_program()

    nodes = list(edge_program.graph.nodes)
    assert not graph_contains_any_of_ops(edge_program.graph, [torch.ops.aten.gru.input])
    delegate_call_index = num_layers * 2 + 7
    assert nodes[delegate_call_index].name == 'executorch_call_delegate'
    # The delegate call produces every GRU's `Y_h` output, and last GRU's `Y` output (num_layers + 1 in total).
    assert len(nodes[delegate_call_index].users.keys()) == num_layers + 1


@pytest.mark.parametrize('num_layers', [2, 3, 5], ids=lambda num_layers: f'num_layers = {num_layers}')
def test_gru_splitting__bidirectional__no_bias(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(num_layers=num_layers, D=2)
    model = GruModule(input_size, hidden_size, num_layers=num_layers, bidirectional=True, bias=False).eval()

    example_input = tuple(torch.ones(input_shape) for input_shape in input_shapes)
    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)

    assert len(exir_program_aten.graph.nodes) == 6 + (num_layers) * 4
    assert len([n for n in exir_program_aten.graph.nodes if 'gru' in n.name]) == 1  # Just 1 `GRU` in the model.

    # Run pre-processing passes of the float32 aten dialect program.
    pytorch_pass_manager = NXPPyTorchPassManager(exir_program_aten, [
        SplitGRUBasedOnNumLayers
    ])
    pytorch_pass_manager.run()

    assert len(exir_program_aten.graph.nodes) == 5 + (num_layers) * 8
    assert len([n for n in exir_program_aten.graph.nodes if 'gru' in n.name]) == num_layers  # Many `GRU` in the model.


@pytest.mark.parametrize('num_layers', [2, 3], ids=lambda num_layers: f'num_layers = {num_layers}')
def test_gru_splitting__bidirectional__no_bias__delegation(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(num_layers=num_layers, D=2)
    model = GruModule(input_size, hidden_size, num_layers=num_layers, bidirectional=True, bias=False).eval()

    edge_program = to_quantized_edge_program(model, to_model_input_spec(input_shapes)).exported_program()

    nodes = list(edge_program.graph.nodes)
    assert not graph_contains_any_of_ops(edge_program.graph, [torch.ops.aten.gru.input])
    delegate_call_index = num_layers * 2 + 7
    assert nodes[delegate_call_index].name == 'executorch_call_delegate'
    # The delegate call produces every GRU's `Y_h` output, and last GRU's `Y` output (num_layers + 1 in total).
    assert len(nodes[delegate_call_index].users.keys()) == num_layers + 1


@pytest.mark.parametrize('num_layers', [2, 3, 7], ids=lambda num_layers: f'num_layers = {num_layers}')
def test_gru_splitting__bidirectional__with_bias(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(num_layers=num_layers, D=2)
    model = GruModule(input_size, hidden_size, num_layers=num_layers, bidirectional=True, bias=True).eval()

    example_input = tuple(torch.ones(input_shape) for input_shape in input_shapes)
    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)

    assert len(exir_program_aten.graph.nodes) == 6 + (num_layers) * 8
    assert len([n for n in exir_program_aten.graph.nodes if 'gru' in n.name]) == 1  # Just 1 `GRU` in the model.

    # Run pre-processing passes of the float32 aten dialect program.
    pytorch_pass_manager = NXPPyTorchPassManager(exir_program_aten, [
        SplitGRUBasedOnNumLayers
    ])
    pytorch_pass_manager.run()

    assert len(exir_program_aten.graph.nodes) == 5 + (num_layers) * 12
    assert len([n for n in exir_program_aten.graph.nodes if 'gru' in n.name]) == num_layers  # Many `GRU` in the model.


@pytest.mark.parametrize('num_layers', [2, 3], ids=lambda num_layers: f'num_layers = {num_layers}')
def test_gru_splitting__bidirectional__with_bias__delegation(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(num_layers=num_layers, D=2)
    model = GruModule(input_size, hidden_size, num_layers=num_layers, bidirectional=True, bias=True).eval()

    edge_program = to_quantized_edge_program(model, to_model_input_spec(input_shapes)).exported_program()

    nodes = list(edge_program.graph.nodes)
    assert not graph_contains_any_of_ops(edge_program.graph, [torch.ops.aten.gru.input])
    delegate_call_index = num_layers * 2 + 7
    assert nodes[delegate_call_index].name == 'executorch_call_delegate'
    # The delegate call produces every GRU's `Y_h` output, and last GRU's `Y` output (num_layers + 1 in total).
    assert len(nodes[delegate_call_index].users.keys()) == num_layers + 1
