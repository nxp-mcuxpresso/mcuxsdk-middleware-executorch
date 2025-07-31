# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from torch import nn
from torch.export import ExportedProgram

from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass_manager import NXPPyTorchPassManager
from executorch.backends.nxp.pytorch_passes.turn_batch_first_gru_to_time_major import TurnBatchFirstGRUToTimeMajor
from executorch.backends.nxp.tests.executorch_pipeline import get_random_calibration_inputs, \
    to_model_input_spec, to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.exir.dialects._ops import ops as exir_ops


class BatchFirstGRUModel(nn.Module):
    def __init__(self, num_layers=1, input_size=8):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size=24, num_layers=num_layers, batch_first=True)

    def forward(self, input_):
        # `input_` has shape [batch_size, sequence_length, input_size]
        return self.gru(input_, None)


class TestTurningGRUToTimeMajor(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    def test_turning_gru_to_time_major__single_layer__aten_pass(self):
        batch_size, input_size, sequence_len = 2, 8, 16
        model = BatchFirstGRUModel(input_size=input_size).eval()
        input_shape = (batch_size, sequence_len, input_size)

        calibration_inputs = get_random_calibration_inputs(to_model_input_spec(input_shape))
        example_input = calibration_inputs[0]

        exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)

        outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Apply the optimization.
        NXPPyTorchPassManager(exir_program_aten, [TurnBatchFirstGRUToTimeMajor]).run()

        nodes = list(exir_program_aten.graph.nodes)

        # Make sure the permute operators were correctly added, and the GRU is now time major.
        gru = nodes[7]
        main_getitem = list(gru.users.keys())[0]
        assert gru.target == torch.ops.aten.gru.input
        assert not gru.args[-1]  # `not batch_first`.
        assert 'getitem' in main_getitem.target.__name__
        assert gru.args[0].target == list(main_getitem.users.keys())[0].target == torch.ops.aten.permute.default

        outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Make sure the model still produces the exact same output.
        assert np.allclose(outputs_before[0], outputs_after[0])
        assert np.allclose(outputs_before[1], outputs_after[1])

    def test_turning_gru_to_time_major__single_layer__delegation(self):
        batch_size, input_size, sequence_len = 2, 8, 16
        model = BatchFirstGRUModel(input_size=input_size).eval()
        input_shape = (batch_size, sequence_len, input_size)

        edge_program = to_quantized_edge_program(model, to_model_input_spec(input_shape)).exported_program()

        nodes = list(edge_program.graph.nodes)
        assert not graph_contains_any_of_ops(edge_program.graph, [torch.ops.aten.gru.input])
        assert nodes[3].target == exir_ops.edge.aten.permute_copy.default  # Unsupported permutation -> not delegated.
        assert nodes[3].args[1] == [1, 0, 2]
        assert nodes[6].name == 'executorch_call_delegate'
        assert nodes[11].target == exir_ops.edge.aten.permute_copy.default  # Unsupported permutation -> not delegated.
        assert nodes[11].args[1] == [1, 0, 2]

    def test_turning_gru_to_time_major__multi_layer__aten_pass(self):
        batch_size, input_size, sequence_len = 2, 8, 16
        model = BatchFirstGRUModel(input_size=input_size, num_layers=2).eval()
        input_shape = (batch_size, sequence_len, input_size)

        calibration_inputs = get_random_calibration_inputs(to_model_input_spec(input_shape))
        example_input = calibration_inputs[0]

        exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)

        outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Apply the optimizations. Use the default list of optimizations to make sure they are applied in the correct
        #  order. First, the permute operators are added, then the multilayer GRU is decomposed.
        NXPPyTorchPassManager(exir_program_aten).run()

        nodes = list(exir_program_aten.graph.nodes)
        targets = [n.target.__name__ for n in nodes if n.op == "call_function"]
        assert targets.count('gru.input') == 2
        assert targets.count('permute.default') == 2  # Just the main IO was transposed, not the decomposed GRU ops.

        # Make sure the permute operators were correctly added, and the GRU is now time major.
        gru1 = nodes[12]
        gru2 = nodes[15]
        gru2_main_getitem = list(gru2.users.keys())[0]
        assert gru1.target == gru2.target == torch.ops.aten.gru.input
        assert not gru1.args[-1]  # `not batch_first`.
        assert not gru2.args[-1]  # `not batch_first`.
        assert gru1.args[0].target == torch.ops.aten.permute.default
        assert 'getitem' in gru2_main_getitem.target.__name__
        assert list(gru2_main_getitem.users.keys())[0].target == torch.ops.aten.permute.default

        outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Make sure the model still produces the exact same output.
        assert np.allclose(outputs_before[0], outputs_after[0])
        assert np.allclose(outputs_before[1], outputs_after[1])

    def test_turning_gru_to_time_major__multi_layer__delegation(self):
        batch_size, input_size, sequence_len = 2, 8, 16
        model = BatchFirstGRUModel(input_size=input_size, num_layers=2).eval()
        input_shape = (batch_size, sequence_len, input_size)

        edge_program = to_quantized_edge_program(model, to_model_input_spec(input_shape)).exported_program()

        nodes = list(edge_program.graph.nodes)
        assert not graph_contains_any_of_ops(edge_program.graph, [torch.ops.aten.gru.input])
        assert nodes[3].target == exir_ops.edge.aten.permute_copy.default  # Unsupported permutation -> not delegated.
        assert nodes[3].args[1] == [1, 0, 2]
        assert nodes[6].name == 'executorch_call_delegate'
        assert len(nodes[6].users.keys()) == 3  # 2 Y_h outputs, 1 Y output. Both GRU nodes were delegated.
        assert nodes[13].target == exir_ops.edge.aten.permute_copy.default  # Unsupported permutation -> not delegated.
        assert nodes[13].args[1] == [1, 0, 2]

    def test_turning_gru_to_time_major__2D_input(self):
        input_size, sequence_len = 8, 16
        model = BatchFirstGRUModel(input_size=input_size).eval()
        input_shape = (sequence_len, input_size)  # Unbatched input.

        calibration_inputs = get_random_calibration_inputs(to_model_input_spec(input_shape))
        example_input = calibration_inputs[0]

        exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)

        outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Apply the optimization.
        NXPPyTorchPassManager(exir_program_aten, [TurnBatchFirstGRUToTimeMajor]).run()

        nodes = list(exir_program_aten.graph.nodes)

        # Make sure the permute operators were correctly added, and the GRU is now time major.
        gru = nodes[8]
        main_getitem = list(gru.users.keys())[0]
        assert nodes[1].target == torch.ops.aten.unsqueeze.default  # Added to make the input 3D.
        assert gru.target == torch.ops.aten.gru.input
        assert not gru.args[-1]  # `not batch_first`.
        assert 'getitem' in main_getitem.target.__name__
        assert gru.args[0].target == list(main_getitem.users.keys())[0].target == torch.ops.aten.permute.default

        outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Make sure the model still produces the exact same output.
        assert np.allclose(outputs_before[0], outputs_after[0])
        assert np.allclose(outputs_before[1], outputs_after[1])
