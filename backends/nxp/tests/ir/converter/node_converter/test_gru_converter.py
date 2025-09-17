# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from parameterized import parameterized

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.exir.dialects._ops import ops as exir_ops


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


def get_gru_input_shapes(
    input_size=8,
    hidden_size=8,
    sequence_length=8,
    batch_size=8,
    num_layers=1,
    D=1,  # Unidirectional.
):
    input_shapes = [
        (batch_size, sequence_length, input_size),  # Flipped dimension order.
        (D * num_layers, batch_size, hidden_size)
    ]

    return input_shapes, input_size, hidden_size, sequence_length


class TestGRUConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    def test_gru__batch_first__rt700__batch_size_1(self):
        input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(
            8, 16, 3,
            batch_size=1
        )
        model = GruModule(
            input_size, hidden_size,
            batch_first=True  # Only supported when `batch_size == 1`.
        ).eval()

        edge_program = to_quantized_edge_program(
            model, input_shapes,
            target='imxrt700'
        ).exported_program()

        # Make sure the GRU was delegated.
        nodes = list(edge_program.graph.nodes)
        assert nodes[5].name == 'executorch_call_delegate'
        assert not graph_contains_any_of_ops(edge_program.graph, [exir_ops.edge.aten.gru.input])

    def test_gru__unsupported_input_size__rt700(self):
        input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(input_size=7)  # Not a multiple of 8.
        model = GruModule(input_size, hidden_size).eval()

        edge_program = to_quantized_edge_program(
            model, input_shapes,
            target='imxrt700'
        ).exported_program()

        # Make sure the GRU was not delegated.
        assert graph_contains_any_of_ops(edge_program.graph, [exir_ops.edge.aten.gru.input])

    def test_gru__unsupported_hidden_size__rt700(self):
        input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(hidden_size=7)  # Not a multiple of 8.
        model = GruModule(input_size, hidden_size).eval()

        edge_program = to_quantized_edge_program(
            model, input_shapes,
            target='imxrt700'
        ).exported_program()

        # Make sure the GRU was not delegated.
        assert graph_contains_any_of_ops(edge_program.graph, [exir_ops.edge.aten.gru.input])

    @parameterized.expand([
        [16, 8, 5, 3],
        [128, 128, 2, 2],
        [8, 32, 4, 5],
    ])
    def test_gru(self, input_size, hidden_size, sequence_length, batch_size):
        num_layers = 1  # Other values are not supported.
        D = 1  # One directional.
        input_shapes = [
            (sequence_length, batch_size, input_size),
            (D * num_layers, batch_size, hidden_size)
        ]
        model = GruModule(input_size, hidden_size=hidden_size, bidirectional=False, bias=True,
                          num_layers=num_layers).eval()

        edge_program = to_quantized_edge_program(model, input_shapes).exported_program()

        # Make sure the GRU was delegated.
        nodes = list(edge_program.graph.nodes)
        assert nodes[5].name == 'executorch_call_delegate'
        assert not graph_contains_any_of_ops(edge_program.graph, [exir_ops.edge.aten.gru.input])

    def test_gru__no_bias(self):
        num_layers = 1
        D = 1  # One directional.
        input_size, hidden_size, sequence_length, batch_size = 16, 8, 5, 3
        input_shapes = [
            (sequence_length, batch_size, input_size),
            (D * num_layers, batch_size, hidden_size)
        ]
        model = GruModule(input_size, hidden_size=hidden_size, bidirectional=False, bias=False,
                          num_layers=num_layers).eval()

        edge_program = to_quantized_edge_program(model, input_shapes).exported_program()

        # Make sure the GRU was delegated.
        nodes = list(edge_program.graph.nodes)
        assert nodes[5].name == 'executorch_call_delegate'
        assert not graph_contains_any_of_ops(edge_program.graph, [exir_ops.edge.aten.gru.input])

    @parameterized.expand([
        [16, 8, 5, 3],
        [8, 32, 4, 5],
    ])
    def test_gru__bidirectional(self, input_size, hidden_size, sequence_length, batch_size):
        num_layers = 1  # Other values are not supported.
        D = 2  # Bidirectional.
        input_shapes = [
            (sequence_length, batch_size, input_size),
            (D * num_layers, batch_size, hidden_size)
        ]
        model = GruModule(input_size, hidden_size=hidden_size, bidirectional=True, bias=True,
                          num_layers=num_layers).eval()

        edge_program = to_quantized_edge_program(model, input_shapes).exported_program()

        # Make sure the GRU was delegated.
        nodes = list(edge_program.graph.nodes)
        assert nodes[5].name == 'executorch_call_delegate'
        assert not graph_contains_any_of_ops(edge_program.graph, [exir_ops.edge.aten.gru.input])

    @parameterized.expand([
        [16, 8, 5, 3],
        [8, 32, 4, 5],
    ])
    def test_gru__bidirectional__no_bias(self, input_size, hidden_size, sequence_length, batch_size):
        num_layers = 1  # Other values are not supported.
        D = 2  # Bidirectional.
        input_shapes = [
            (sequence_length, batch_size, input_size),
            (D * num_layers, batch_size, hidden_size)
        ]
        model = GruModule(input_size, hidden_size=hidden_size, bidirectional=True, bias=False,
                          num_layers=num_layers).eval()

        edge_program = to_quantized_edge_program(model, input_shapes).exported_program()

        # Make sure the GRU was delegated.
        nodes = list(edge_program.graph.nodes)
        assert nodes[5].name == 'executorch_call_delegate'
        assert not graph_contains_any_of_ops(edge_program.graph, [exir_ops.edge.aten.gru.input])
