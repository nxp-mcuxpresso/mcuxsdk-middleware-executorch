# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.export.unflatten import _AttrKind, _assign_attr
from torch.fx import Node

from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass import NXPPyTorchPass


class RemoveNodesWithKnownOutputs(NXPPyTorchPass):
    """ In some situations, a node will always produce the same output data at runtime. If these cases are identified,
         the nodes can simply be removed and replaced by a static parameter node, which holds the data the original
         node would produce.
        This pass identifies some of these cases and performs the replacement.
    """

    # Nodes which don't have the `.meta['val']` attribute. The datatype and shape of their inferred output data will
    #  therefore not be checked against the expected values in the `.meta['val']`.
    nodes_without_val_meta = [
        torch.ops.aten.empty.memory_format,
        operator.getitem  # Generic, but required to remove the `torch.split` node.
    ]

    def replace_nodes_in_list_with_their_data(self, list_of_args: list) -> list | None:
        """ Replace the nodes in `list_of_orgs` by their static data. If not all data is available, return `None`.

        :param list_of_args: List of arguments of an aten operator. Can include nodes, generic arguments, lists...
        :return:`list_of_args` but with tensors replaced by their static data, or `None` if not all data is available.
        """
        args_with_data = []
        for arg in list_of_args:
            match arg:
                case Node():
                    # `arg` is either another operator, a model input, or a static parameter.
                    data = self.get_tensor_constant_from_node(arg)
                    if data is None:
                        # No static data is available.
                        return None

                    args_with_data.append(data)
                case list():
                    nested = self.replace_nodes_in_list_with_their_data(arg)
                    if nested is None:
                        return None
                    args_with_data.append(nested)

                case _:
                    # Generic argument. Not an input from a previous node.
                    args_with_data.append(arg)

        return args_with_data

    @staticmethod
    def node_is_followed_only_by_getitem_nodes(node: Node) -> bool:
        def _is_getitem(node_: Node) -> bool:
            return node_.op == "call_function" and node_.target.__name__ == "getitem"

        users = list(node.users.keys())
        return all(_is_getitem(user) for user in users)

    def replace_node_with_static_data(self, node: Node, static_data: torch.Tensor):
        """ Remove the given `node` from the graph and replace it with a parameter node containing the `static_data`.
        """
        # Generate a unique name for the new static parameter.
        new_name = get_new_attr_name_with_prefix(node.name)(self.module)

        # Create the node for the parameter.
        param = torch.nn.Parameter(static_data, False)
        _assign_attr(param, self.module, str(new_name), _AttrKind.PARAMETER)
        with self.module.graph.inserting_before(node):
            static_parameter_node = self.module.graph.get_attr(new_name)

        # Replace the old node with the new static parameter.
        node.replace_all_uses_with(static_parameter_node)
        self.module.graph.erase_node(node)

    def replace_following_getitem_nodes_with_static_data(
        self,
        root_node: Node,
        static_data: tuple[torch.Tensor, ...]
    ) -> bool:
        """ Remove the `root_node` and all `GetItem` nodes that consume its output from the graph, and replace their
             uses with parameter nodes containing the provided `static_data`.
            If something other than just `GetItem` nodes follow after the `root_node`, nothing is done and `False` is
             returned.

        :param root_node: The main compute node which is followed only by `GetItem` nodes.
        :param static_data: A tuple of static tensors with the data that will be used to replace the `GetItem` nodes
                             after the `root_node`.
        :return: `True` if the replacement was successfully executed. `False` otherwise.
        """

        if not self.node_is_followed_only_by_getitem_nodes(root_node):
            return False  # Unexpected case.

        users = list(root_node.users.keys())
        if len(users) != len(static_data):
            return False  # Unexpected missmatch.

        # Replace the individual `GetItem` nodes.
        for get_item_node in users:
            idx = get_item_node.args[1]
            self.replace_node_with_static_data(get_item_node, static_data[idx])

        # Finally remove the root node from the graph.
        self.module.graph.erase_node(root_node)

        return True

    def data_matches_node_meta(self, node: Node, data: torch.tensor) -> bool:
        """ Verify that the provided `data` tensor has the same shape and datatype as the `node`. """
        if node.target not in self.nodes_without_val_meta:
            if node.meta['val'].shape != data.shape:
                return False  # The inferred data has a different shape than expected.

            if node.meta['val'].dtype != data.dtype:
                return False  # The inferred data has a different data type than expected.

        return True

    def data_matches_meta_of_following_getitem_nodes(self, root_node: Node, data: tuple[torch.Tensor, ...]) -> bool:
        """ Verify that the provided `data` tensor has the same shape and datatype as the `GetItem` nodes which consume
             the output of the `root_node`.
        """
        if not self.node_is_followed_only_by_getitem_nodes(root_node):
            return False  # Unexpected case

        users = list(root_node.users.keys())
        return all(self.data_matches_node_meta(get_item, data[get_item.args[1]]) for get_item in users)

    def run(self) -> bool:
        for node in self.module.graph.nodes:
            if not node.op == "call_function":
                continue  # Not a compute operator.

            # Try to access the static data for the inputs of the node.
            args_with_data = self.replace_nodes_in_list_with_their_data(node.args)

            if args_with_data is None:
                # Output data inference is not possible.
                continue

            # All input data is static. Run the operator to compute the input it would produce at runtime.
            # noinspection PyBroadException
            try:
                output = node.target(*args_with_data, **node.kwargs)

                if isinstance(output, tuple):
                    if not self.data_matches_meta_of_following_getitem_nodes(node, output):
                        continue  # The inferred data does not have the expected type/shape.
                else:
                    if not self.data_matches_node_meta(node, output):
                        continue  # The inferred data does not have the expected type/shape.

            except Exception:
                continue  # Failed to infer the data. Continue with the other nodes.

            # The output data appears to have been correctly inferred. Create a static parameter node for it.

            if isinstance(output, tuple):
                # The node produces multiple outputs (e.g. `split`). If the node is followed only by `GetItem` nodes
                #  which extract the individual outputs, replace them by the static data.
                return self.replace_following_getitem_nodes_with_static_data(node, output)

            else:
                self.replace_node_with_static_data(node, output)
                return True  # Indicate that changes were made.

        return False  # Nothing changed.
