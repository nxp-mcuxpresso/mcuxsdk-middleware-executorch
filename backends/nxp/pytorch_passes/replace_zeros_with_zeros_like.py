# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from torch.fx import Node

from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass import NXPPyTorchPass


class ReplaceZerosWithZerosLikePass(NXPPyTorchPass):
    """ PyTorch currently does not accept nodes with keyword arguments, with some exceptions, e.g. zeros_like [1].
        To be able to quantize aten.zeros, a conversion to aten.zeros_like is necessary.

                                                                            │
                                                               ┌────────────▼─────────────┐
                                                               │ aten.empty.memory_format │
                                                               └────────────┬─────────────┘
                       │                                                    │
             ┌─────────▼──────────┐        replace with        ┌────────────▼────────────┐
             │ aten.zeros.default │       ──────────────►      │ aten.zeros_like.default │
             └─────────┬──────────┘                            └────────────┬────────────┘
                       ▼                                                    ▼

        [1] https://github.com/pytorch/pytorch/blob/v2.6.0/torch/ao/quantization/pt2e/prepare.py#L437-L442
    """

    def run(self) -> bool:
        def _is_zeros(node_: Node) -> bool:
            return node_.op == "call_function" and node_.target == torch.ops.aten.zeros.default

        made_changes = False

        if not any(map(_is_zeros, self.module.graph.nodes)):
            return made_changes  # No zeros nodes in the model.

        for node in self.module.graph.nodes:
            if not _is_zeros(node):
                continue  # Not zeros node.

            zeros_node = node

            with self.module.graph.inserting_before(zeros_node):
                empty_memory_format = self.module.graph.call_function(torch.ops.aten.empty.memory_format, (zeros_node.args[0],))

            with self.module.graph.inserting_before(zeros_node):
                zeros_like_node = self.module.graph.call_function(torch.ops.aten.zeros_like.default, (empty_memory_format,),
                                                                  kwargs={"pin_memory": zeros_node.kwargs["pin_memory"]})

            # Copy zeros node meta
            zeros_val = zeros_node.meta["val"]
            del zeros_node.meta["val"]
            zeros_like_node.meta = copy.deepcopy(zeros_node.meta)
            zeros_like_node.meta["val"] = zeros_val.detach().clone()

            # Replace the uses of the zeros with the zeros_like.
            zeros_node.replace_all_uses_with(zeros_like_node)

            self.module.graph.erase_node(zeros_node)

            made_changes = True

        return made_changes
