# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch._subclasses import FakeTensorMode, FakeTensor
from torch.fx import Node

from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass import NXPPyTorchPass


class TurnBatchFirstGRUToTimeMajor(NXPPyTorchPass):
    """ Replace a `batch_first` `aten.gru.input` operator with an equivalent time major version of it, by permuting its
         inputs. This is done because Neutron only supports the time major GRU.
        Ideally, this pass should be executed before `split_gru_based_on_num_layers`, in order to minimize the number
         of added `permute_copy` operators.

        The use case is illustrated below. The tensor shapes annotations use:
            - N (batch size)
            - L (sequence length)
            - D (2 if bidirectional else 1)
            - Hin (input size)
            - Hout (hidden size)

                                                                  │ [N, L, Hin]
                                                          ┌───────▼───────┐
                                                          │  PermuteCopy  │
                                                          │  perm=(1,0,2) │
                                                          └───────┬───────┘
                             │ [N, L, Hin]                        │ [L, N, Hin]
                   ┌─────────▼─────────┐                ┌─────────▼─────────┐
                   │        GRU        │  replace with  │        GRU        │
                   │ batch_first=True  │  ───────────►  │ batch_first=False │
                   └─────────┬─────────┘                └─────────┬─────────┘
                             ▼ [N, L, D*Hout]                     │ [L, N, D*Hout]
                                                          ┌───────▼───────┐
                                                          │  PermuteCopy  │
                                                          │  perm=(1,0,2) │
                                                          └───────┬───────┘
                                                                  ▼ [N, L, D*Hout]
    """

    def _create_permute_node_after(self, prev_node: Node, perm: tuple[int, ...], set_input: bool = True) -> Node:
        with self.module.graph.inserting_after(prev_node):
            permute_node = self.module.graph.call_function(
                torch.ops.aten.permute.default,
                (prev_node, perm) if set_input else (None, perm)
            )

        # Assign the `source_fn_stack` and `val` meta fields as they are required for quantization.
        permute_node.meta['source_fn_stack'] = [(permute_node.name, torch.permute)]

        # Compute the output shape for the `permute`, and assign the `val` meta.
        input_val = prev_node.meta['val']
        with FakeTensorMode() as mode:
            fake_input = FakeTensor.from_tensor(torch.empty(input_val.shape, dtype=input_val.dtype), mode)
            permute_node.meta['val'] = torch.permute(fake_input, perm)

        return permute_node

    def run(self) -> bool:
        def _is_gru(node_: Node) -> bool:
            return node_.op == "call_function" and node_.target == torch.ops.aten.gru.input

        for node in self.module.graph.nodes:
            if not _is_gru(gru := node):
                continue  # Not GRU.

            batch_first = gru.args[-1]
            if not batch_first:
                continue  # Already time major.

            gru_main_input = gru.args[0]
            main_output_getitem = list(gru.users.keys())[0]
            if not (len(gru_main_input.meta['val'].shape) == len(main_output_getitem.meta['val'].shape) == 3):
                continue  # Should never happen.

            # The first input and output must be permuted using the [1, 0, 2] permutation. Add the permute ops.
            permute_node_before = self._create_permute_node_after(gru_main_input, (1, 0, 2))
            # Don't set the input for the trailing permute node yet (set_input=False), because it would get replaced by
            #  `replace_all_uses_with`.
            permute_node_after = self._create_permute_node_after(main_output_getitem, (1, 0, 2), set_input=False)
            main_output_getitem.replace_all_uses_with(permute_node_after)
            permute_node_after.args = (main_output_getitem,) + permute_node_after.args[1:]  # Now set the input.

            # Set `batch_first` to `False` and set the GRU's input to the first `permute` node.
            gru.args = (permute_node_before,) + gru.args[1:-1] + (False,)

            # One instance was transformed. Do not continue traversing the graph for other instances, as the list of
            #  nodes that the loop iterates over has been changed. Return `True` to indicate the change, and the pass
            #  manager will call this pass again, which will transform any applicable following GRU nodes.
            return True

        return False
