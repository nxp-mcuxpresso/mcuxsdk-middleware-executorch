# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Partitioner for the NXP Neutron NPU

import collections
import itertools
import logging
import operator
from copy import copy
from dataclasses import dataclass
from typing import final, Optional, Mapping

import torch
from torch.export.exported_program import ExportedProgram
from torch.fx.node import _get_qualified_name, Node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.nn import Parameter

from executorch.backends.nxp.backend.custom_delegation_options import CustomDelegationOptions
from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.backend.ir.converter.node_converter import Target
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import *
from executorch.backends.nxp.nxp_backend import NeutronBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects._ops import ops as exir_ops


class QDQClusterRecognizer:
    """
    Implementation of the Quantize - Dequantize clustering.
    The quantization is captured in the ExecuTorch program using the QDQ (Quantize - DeQuantize) representation. Here
    the inputs to a node comes from some dequantize nodes and outputs goes to some quantize nodes.
    The QDQClusterRecognizer identifies operator performing the quantized arithmetic represented in QDQ form, and the
    corresponding QDQ cluster. The QDQ cluster consists of the:
    - dequantize nodes producing the inputs to the compute node
    - compute node (e.g. conv)
    - auxiliary nodes, like getitem, view_copy, ... which does not perform a core computation
    - quantize nodes processing the output of the compute node.
    """

    @dataclass
    class QDQCluster:
        """
        Dataclass to hold the QDQ cluster instance. For the purpose of Partitioner we hold the list of operators,
        in the QDQ cluster (`ops`) and the compute node what the QDQ cluster is built around.
        The compute node is what is represented in the Neutron IR. the rest of nodes are helpers for data transformation,
        and defines the quantization parameters. This gives the partitioner the ability to:
            - identify if the node is part of a QDQ cluster
            - reference the compute node in the QDQ cluster
        """
        compute_node: torch.fx.Node
        ops: list[torch.fx.Node]

    QUANTIZE_OPERATORS = [
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    ]

    DEQUANTIZE_OPERATORS = [
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    ]

    AUXILIARY_OPS = [
        operator.getitem,
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.clone.default,
    ]

    def __init__(self):
        self.cluster_map: dict[str, QDQClusterRecognizer.QDQCluster] = {}

    @staticmethod
    def is_quant_node(node: torch.fx.Node) -> bool:
        return node.target in QDQClusterRecognizer.QUANTIZE_OPERATORS

    @staticmethod
    def is_dequant_node(node: torch.fx.Node) -> bool:
        return node.target in QDQClusterRecognizer.DEQUANTIZE_OPERATORS

    @staticmethod
    def is_auxiliary_node(node: torch.fx.Node) -> bool:
        return node.target in QDQClusterRecognizer.AUXILIARY_OPS

    def get_qdq_cluster_input_part(self, node: torch.fx.Node) -> list[torch.fx.Node]:
        """
        Return the list of nodes representing the input part of the QDQ cluster of the node `node`.
        Those are various dequantization nodes (see DEQUANTIZE_OPERATORS) optionally followed by auxiliary
        nodes.
        If the `node` not meets the QDQ cluster schema, returns empty list.
        """

        # Iterative search for input nodes of the QDQ Cluster:
        nodes_to_check = [node]
        qdq_cluster = []
        while len(nodes_to_check) > 0:
            n = nodes_to_check.pop()
            qdq_cluster.append(n)
            if self.is_dequant_node(n):
                continue
            input_nodes_from_dequant_or_helper = [(self.is_dequant_node(i) or self.is_auxiliary_node(i))
                                                  for i in n.all_input_nodes]
            if all(input_nodes_from_dequant_or_helper):
                nodes_to_check.extend(n.all_input_nodes)
            else:
                return []

        logging.debug(f"Dequant Cluster for {node} is: {qdq_cluster}")
        return qdq_cluster

    def get_qdq_cluster_output_part(self, node: torch.fx.Node) -> list[torch.fx.Node]:
        """
        Returns the list of nodes representing the output part of the QDQ cluster of the `node`.
        Those are various quantize nodes (see QUANTIZE_OPERATORS) preceded by auxiliary nodes.
        If the `node` not meets the QDQ cluster schema, returns empty list.
        """

        # Iterative search for output nodes of the QDQ Cluster:
        nodes_to_check = [node]
        qdq_cluster = []
        while len(nodes_to_check) > 0:
            n = nodes_to_check.pop()
            qdq_cluster.append(n)
            if self.is_quant_node(n):
                continue
            consumers = [ngn for ngn in list(node.graph.nodes) if n in ngn.all_input_nodes]
            logging.debug(f"\t Users for node {n} are: {consumers}")
            output_nodes_to_quant_or_helper = [(self.is_quant_node(i) or self.is_auxiliary_node(i))
                                               for i in consumers]
            if all(output_nodes_to_quant_or_helper):
                nodes_to_check.extend(consumers)
            else:
                return []

        logging.debug(f"Quant Cluster for {node} is {qdq_cluster}")
        return qdq_cluster

    def get_qdq_cluster(self, node: torch.fx.Node) -> list[torch.fx.Node]:
        """
        Returns the QDQ cluster of the operator, if quantized. If operator is not quantized, returns empty list.
        """
        logging.debug(node)
        input_qdq_cluster = self.get_qdq_cluster_input_part(node)
        output_qdq_cluster = self.get_qdq_cluster_output_part(node)
        if input_qdq_cluster and output_qdq_cluster:
            return list(set(input_qdq_cluster).union(output_qdq_cluster))
        else:
            return []

    def tag_nodes(self, nodes: list[torch.fx.Node], cluster_name: str) -> None:
        """
        Tags a node and its related dequant and quant nodes with a specified cluster name
        """
        for node in nodes:
            logging.info(f"Tagging node {node} as {cluster_name}")
            node.meta["cluster"] = cluster_name

    def tag_qdq_clusters(self, nodes: list[torch.fx.Node]):
        """
        Identifies QDQ clusters and tag them based on compute operation inside.
        """

        for node in nodes:
            if (node.op == "call_function" and
                not self.is_quant_node(node) and
                not self.is_dequant_node(node)):
                cluster = self.get_qdq_cluster(node)
                if cluster:
                    cluster_name = f"{node.name}_cluster"
                    self.tag_nodes(cluster, cluster_name)
                    self.cluster_map[cluster_name] = self.QDQCluster(node, cluster)


supported_ops = {
    exir_ops.edge.aten.addmm.default: AddMMConverter,
    exir_ops.edge.aten.avg_pool2d.default: AvgPool2dConverter,
    exir_ops.edge.aten.constant_pad_nd.default: ConstantPadNDConverter,
    exir_ops.edge.aten.convolution.default: ConvolutionConverter,
    exir_ops.edge.aten.max_pool2d.default: MaxPool2dConverter,
    exir_ops.edge.aten.max_pool2d_with_indices.default: MaxPool2dConverter,
    exir_ops.edge.aten.mm.default: MMConverter,
    exir_ops.edge.aten.relu.default: ReLUConverter,
    exir_ops.edge.aten.hardtanh.default: HardTanhConverter,
    exir_ops.edge.aten.tanh.default: TanhConverter,
    exir_ops.edge.aten._softmax.default: SoftmaxConverter,
    exir_ops.edge.aten.view_copy.default: ViewCopyConverter,
    exir_ops.edge.aten.add.Tensor: AddTensorConverter,
    exir_ops.edge.aten.mean.dim: MeanDimConverter,
    exir_ops.edge.aten._adaptive_avg_pool2d.default: AdaptiveAvgPool2dConverter,
    exir_ops.edge.aten.clone.default: CloneConverter,
    exir_ops.edge.aten.abs.default: AbsConverter,
    exir_ops.edge.aten.cat.default: CatConverter,
    exir_ops.edge.aten.sigmoid.default: SigmoidConverter,
    exir_ops.edge.aten.gru.input: GRUConverter,
    exir_ops.edge.aten.permute_copy.default: PermuteCopyConverter,
}


class NeutronSupportedOperators(OperatorSupportBase):

    def __init__(
        self,
        qdq_clusters: dict[str, QDQClusterRecognizer.QDQCluster],
        target: Target,
        operators_not_to_delegate: list[str],
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions
    ):
        self.qdq_clusters = qdq_clusters
        self.target = target
        self.operators_not_to_delegate = operators_not_to_delegate
        self.parameters_mapping = parameters_mapping
        self.custom_delegation_options = custom_delegation_options

    def _is_node_quantized(self, node: torch.fx.node.Node):
        return "cluster" in node.meta

    def _is_node_call_function(self, node: torch.fx.node.Node):
        return node.op == "call_function"

    def is_node_delegatable(self, node: torch.fx.node.Node):
        if self.operators_not_to_delegate != ['']:
            any_non_delegatable = any(x in node.name for x in self.operators_not_to_delegate)
            return not any_non_delegatable
        return True

    def _is_node_supported_compute(self, node: torch.fx.node.Node) -> bool:
        """
        Operator checking function for compute nodes.
        """
        if not self.is_node_delegatable(node):
            return False

        if (node_converter := supported_ops.get(node.target, None)) is None:
            # There is no `NodeConverter` for this `node`.
            return False

        return (
            self._is_node_call_function(node) and
            self._is_node_quantized(node) and

            # TODO: `view_copy` node should be delegated only if it's not the only operator in the cluster.
            node_converter.is_supported(node, self.target, self.parameters_mapping, self.custom_delegation_options)
        )

    def _is_node_supported_non_compute(self, node: torch.fx.node.Node) -> bool:
        """
        If the node is a quantize, dequantize or auxiliary node inside a QDQ cluster, the support on Neutron
        is determined by the support of the compute operator.
        """
        return (self._is_node_quantized(node) and
                self._is_node_supported_compute(self.qdq_clusters[node.meta["cluster"]].compute_node))

    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        """
        Check if the Edge operator is supported on Neutron.
        """

        if (QDQClusterRecognizer.is_quant_node(node) or
            QDQClusterRecognizer.is_dequant_node(node) or
            QDQClusterRecognizer.is_auxiliary_node(node)):
            return self._is_node_supported_non_compute(node)
        else:
            return self._is_node_supported_compute(node)


class NeutronCapabilityBasedPartitioner(CapabilityBasedPartitioner):
    """
    CapabilityBasedPartitioner with correct partitioning of getitem and its quantize user nodes.

    NeutronCapabilityBasedPartitioner solves problem with not assigning quantize nodes to correct partitions.
    These quantize nodes are consumers of getitem nodes, which need to be assigned to previous (earlier in a graph)
    partition.
    """

    def __is_node_supported(self, node: Node) -> bool:
        return self.operator_support.is_node_supported(
            dict(self.graph_module.named_modules()), node
        )

    def propose_partitions(self) -> list[Partition]:
        # partition_map is a mapping from partition id to a set of partition id's.
        # The value set contains all the partition ids that can be reached by doing a
        # DFS starting from the partition id in the key.
        partition_map: dict[int, set] = collections.defaultdict(set)

        # assumptions: nodes in candidate list is sorted in topological order
        assignment: dict[Node, int] = {}  # mapping from node to partition_id
        partitions_by_id: dict[
            int, Partition
        ] = {}  # mapping from partition_id to partition
        nodes_order: dict[
            Node, int
        ] = {}  # mapping from nodes to reversed topological order
        partitions_order: dict[
            int, int
        ] = {}  # mapping from partition_id to minimum topo order of nodes in partition
        new_partition_id = itertools.count()

        # try to merge partition other_id into partition self_id
        # merge only happens if the end graph doesn't contain cyclic dependency
        # returns `True` when merge happens, `False` otherwise.
        def maybe_merge_partition(self_id: int, other_id: int):
            # merged_nodes is the union of nodes in two partition to-be-merged
            merged_nodes = copy(partitions_by_id[self_id].nodes)
            merged_nodes.update(partitions_by_id[other_id].nodes)

            def dfs_iter_find_cycle(all_user_nodes: set[Node]):
                for user_node in all_user_nodes:
                    visited_partition_ids = set()

                    for path_node in self.dependency_viewer.downstreams_of(user_node):
                        # If any of the nodes in the dfs path of this node are in the merged_nodes
                        # list then there is a cycle in the graph.
                        if path_node in merged_nodes:
                            return True

                        # If any of the nodes in the dfs path of this node are in the assignment
                        # map then we have to make sure that the partitions that these nodes belong
                        # to do not form a cycle with the current partitions being merged. This means
                        # iterating through all the nodes in all the parititons that are traversed in
                        # the dfs path and checking if they are in the merged_nodes list.
                        if path_node in assignment:
                            partition_id = assignment[path_node]
                            # If the partition id has already been visited then we know that it doesn't
                            # form a cycle with the current partitions being merged.
                            if partition_id in visited_partition_ids:
                                continue
                            p_map = partition_map[partition_id]
                            if self_id in p_map or other_id in p_map:
                                return True

                            visited_partition_ids.add(partition_id)

                return False

            # check if merge would create cyclic dependency.
            all_user_nodes = set()
            for node in merged_nodes:
                for user_node in node.users:
                    if user_node not in merged_nodes:
                        all_user_nodes.add(user_node)

            if dfs_iter_find_cycle(all_user_nodes):
                # return false indicating cyclic dependency found and
                # merge is aborted
                return False

            # no cyclic dependency found, move forward with the merge
            # updating partition nodes
            partitions_by_id[self_id].nodes = merged_nodes
            # updating assignment map
            for node in partitions_by_id[other_id].nodes:
                assignment[node] = self_id
            # delete other partition
            del partitions_by_id[other_id]

            partitions_order[self_id] = min(
                partitions_order[self_id], partitions_order[other_id]
            )
            del partitions_order[other_id]

            partition_map[self_id] = partition_map[self_id].union(
                partition_map[other_id]
            )
            del partition_map[other_id]

            return True

        def merge_single_node(node: Node, id: Optional[int]):
            def _update_partition_map(node: Node, id: int):
                # Iterate through all the users of this node and update the partition map to indicate
                # that there is a path from the partition id of this node to the target partition id.
                for user_node in node.users:
                    target_id = assignment.get(user_node, None)
                    if target_id is not None:
                        partition_map[id].add(target_id)
                        partition_map[id].update(partition_map[target_id])

                # Iterate through all the upstream nodes of this node and update the partition map
                # to indicate that there is a path from the partition id of the upstream node to the
                # current node's partition id.
                upstream_nodes = self.dependency_viewer.upstreams_of(node)
                for curr_node in upstream_nodes:
                    source_id = assignment.get(curr_node, None)
                    if source_id is not None:
                        partition_map[source_id].add(id)

            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            if id is None:
                assignment.pop(node)
            elif id not in partitions_by_id:
                assignment[node] = id
                partitions_by_id[id] = Partition(id=id, nodes=[node])
                _update_partition_map(node, id)
            else:
                assignment[node] = id
                partitions_by_id[id].add_node(node)
                _update_partition_map(node, id)

        # MODIFIED PART START
        def get_getitem_and_quantize_users(node: Node):
            # Get a list of node users which are a getitem. If the getitem has following quantize user node, add to list
            # too as partition has to be ended by quantize nodes to form QDQ cluster.
            getitem_and_quantize_users = []
            for user in node.users:
                if (
                    user.op == "call_function"
                    and _get_qualified_name(user.target) == "_operator.getitem"
                ):  # type: ignore[arg-type]
                    getitem_and_quantize_users.append(user)
                    for following_node in user.users:
                        if (
                            following_node.target in QDQClusterRecognizer.QUANTIZE_OPERATORS
                        ):  # type: ignore[arg-type]
                            getitem_and_quantize_users.append(following_node)

            return getitem_and_quantize_users
        # MODIFIED PART END

        logging.debug("Proposing partitions...")

        for node in reversed(self.graph_module.graph.nodes):
            # use dict as an ordered set to ensure deterministic partitioning result, don't care value
            merge_candidates: dict[int, None] = {}

            # Note a limited horizontal fusion is enabled:
            #   when `node` is not supported, the code below attempts to fuse consumer of `node`.
            #
            # I don't see a need to add a knob to disable horizontal fusion yet, we can short-cut
            # the fusion by adding an `else` block here to skip horizontal fusion.
            if self.__is_node_supported(node) and node not in assignment:
                partition_id = next(new_partition_id)
                nodes_order[node] = partition_id
                partitions_order[partition_id] = partition_id
                merge_single_node(node, partition_id)
                merge_candidates[partition_id] = None

            # merge all possible partitions
            for partition_id, _ in sorted(
                partitions_order.items(), key=lambda item: item[1]
            ):
                merge_candidates[partition_id] = None

            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    # note: merge partition `other_id` into partition `self_id` if
                    # it doesn't create cyclic dependency in the graph, otherwise,
                    # this is a no-op
                    maybe_merge_partition(self_id, other_id)

        # MODIFIED PART START
        # post-processing to re-assign "getitem" nodes and following quantize node into upstream partition
        logging.debug("Reassigning getitem nodes and following quantize node to its producer node's partition...")
        for node in self.graph_module.graph.nodes:
            id = assignment.get(node, None)  # type: ignore[arg-type]
            for getitem_quantize in get_getitem_and_quantize_users(node):
                if assignment.get(getitem_quantize, None) != id:  # type: ignore[arg-type]
                    merge_single_node(getitem_quantize, id)
        # MODIFIED PART END

        # filter out single node partitions
        if not self.allows_single_node_partition:
            logging.debug("Filtering out single node partitions...")
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            partitions_to_remove: list[int] = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function":
                        assert callable(node.target)
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if (
                            _get_qualified_name(node.target)
                            in self.allowed_single_node_partition_ops
                        ):
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        logging.debug("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            logging.debug(
                "partition #%s: %s", id, [node.name for node in partition.nodes]
            )

        return [
            partition for partition in partitions_by_id.values() if partition.size() > 0
        ]


@final
class NeutronPartitioner(Partitioner):
    def __init__(
        self,
        compile_spec: list[CompileSpec],
        custom_delegation_options: CustomDelegationOptions | None = None
    ) -> None:
        self.delegation_spec = DelegationSpec(NeutronBackend.__name__, compile_spec)
        self.custom_delegation_options = custom_delegation_options or CustomDelegationOptions()

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the NeutronCapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logging.info("NeutronPartitioner::partition")
        partition_tags = {}

        graph_module = exported_program.graph_module
        nodes = list(graph_module.graph.nodes)

        qdq_clusterer = QDQClusterRecognizer()

        qdq_clusterer.tag_qdq_clusters(nodes)

        graph_module.recompile()
        target = self.delegation_spec[1][2].value
        target = Target(target.decode())

        operators_not_to_delegate = self.delegation_spec[1][4].value.decode().split(',')
        logging.info(f"Operators not to delegate: {operators_not_to_delegate}")

        parameters_mapping = EdgeProgramToIRConverter.map_inputs_to_parameters(exported_program)
        capability_partitioner = NeutronCapabilityBasedPartitioner(
            exported_program.graph_module,
            NeutronSupportedOperators(
                qdq_clusterer.cluster_map,
                target,
                operators_not_to_delegate,
                parameters_mapping,
                self.custom_delegation_options
            ),
            allows_single_node_partition=True,
        )

        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

        tag_constant_data(exported_program)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
