# mypy: ignore-errors
from __future__ import annotations

import sys
from typing import Union, Any
from collections import Counter

from graphviz import Digraph

from .computation_node import NodeContainer
from .computation_node import TensorNode, ModuleNode, FunctionNode

COMPUTATION_NODES = Union[TensorNode, ModuleNode, FunctionNode]

# TODO: Currently, we only use directed graphviz graph since DNN are
# graphs except for e.g. graph neural network (GNN). Experiment on GNN
# and see if undirected graphviz graph can be used to represent GNNs

# TODO: change api of to include also function calls, not only pytorch models
# so, keep the api here as general as possible

# TODO: during traversing graph skip submodules
# if current module is matches depth limit. This will decrease the recursion
# depth of traverse_graph function

node2color = {
    TensorNode: "lightyellow",
    ModuleNode: "darkseagreen1",
    FunctionNode: "aliceblue",
}


class ComputationGraph:
    '''A class to represent Computational graph and visualization of pytorch model

    Attributes:
        visual_graph (Digraph):
            Graphviz.Digraph object to represent computational graph of
            pytorch model

        root_container (NodeContainer):
            Iterable of TensorNodes to represent all input/root nodes
            of pytorch model.

        show_shapes (bool):
            Whether to show shapes of tensor/input/outputs

        hide_inner_tensors (bool):
            Whether to hide inner tensors in graphviz graph object
    '''
    def __init__(
        self,
        visual_graph: Digraph,
        root_container: NodeContainer[TensorNode],
        show_shapes: bool = False,
        hide_inner_tensors: bool = True,
        roll: bool = True,
        depth: int | float = 3,
    ):
        '''
        Resets the running_id, id_dict when a new ComputationGraph is initialized.
        Otherwise, labels would depend on previous ComputationGraph runs
        '''
        self.visual_graph = visual_graph
        self.root_container = root_container
        self.show_shapes = show_shapes
        self.hide_inner_tensors = hide_inner_tensors
        self.roll = roll
        self.depth = depth

        self.reset_id_config()

    def reset_id_config(self):
        '''Resets to id config to the setting of empty visual graph
        needed for getting reproducible/deterministic node name and
        graphviz graphs. This is especially important for output tests
        '''
        self.running_id: int = 0
        self.id_dict: dict[str, int] = {}
        self.edge_list: list[tuple[COMPUTATION_NODES, COMPUTATION_NODES]] = []

    def fill_visual_graph(self):
        '''Fills the graphviz graph with desired nodes and edges.'''

        # First add input nodes
        for root_node in self.root_container:
            root_node.name = 'input-tensor'
            self.add_node(root_node)
            assert len(root_node.children) == 1
            main_module_node = next(iter(root_node.children))
            self.collect_graph(root_node, main_module_node)

        # continue traversing from node of main module
        with RecursionDepth(limit=2000):
            self.traverse_graph(
                main_module_node, main_module_node, self.depth
            )

        self.write_edge()
        self.resize_graph()

    def write_edge(self):
        '''Adds edges to graphviz graph using node ids from edge_list'''
        counter_edge = {}
        for tail, head in self.edge_list:
            tail_id, head_id = self.id_dict[tail.node_id], self.id_dict[head.node_id]
            counter_edge[(tail_id, head_id)] = (
                counter_edge.get((tail_id, head_id), 0) + 1
            )
            self.add_edge(
                tail_id, head_id, counter_edge[(tail_id, head_id)]
            )

    def traverse_graph(
        self,
        node_match: COMPUTATION_NODES,
        start: COMPUTATION_NODES,
        depth_limit: int | float = float('inf'),
        visited: set[COMPUTATION_NODES] | None = None,
    ) -> None:
        '''Use DFS-type traversing to add nodes and edges to graphviz Digraph
        object with additional constrains, e.g. depth limit'''
        if visited is None:
            visited = set()

        if start in visited:
            return

        visited.add(start)

        for node in start.children:
            if node.depth > depth_limit:
                # output tensor that are deeper than depth_limit
                if not node.children:
                    assert isinstance(node, TensorNode), (
                        f"{node} node has no output, "
                        f"it has to be output tensor"
                    )
                    self.collect_graph(node_match, node)
                else:
                    self.traverse_graph(
                        node_match, node, depth_limit, visited
                    )

            # non-deeper nodes: non-tensor nodes or output tensors
            elif (
                not self.hide_inner_tensors or
                (not isinstance(node, TensorNode) or not node.children)
            ):
                self.collect_graph(node_match, node)
                self.traverse_graph(node, node, depth_limit, visited)
            else:
                assert not isinstance(start, TensorNode), (
                    f"{node} node is tensor node and cannot be"
                    f"children of another tensor node {start}"
                )
                self.traverse_graph(
                    node_match, node, depth_limit, visited
                )

    def collect_graph(
        self, tail_node: COMPUTATION_NODES, head_node: COMPUTATION_NODES
    ) -> None:
        '''Adds edges and nodes with appropriate node name/id (so it respects
        properties e.g. if rolled recursive nodes are given the same node name
        in graphviz graph)'''
        if (
            isinstance(tail_node, (FunctionNode, ModuleNode))
            and '-' not in tail_node.node_id
        ):
            if self.roll:
                # identify recursively used modules
                # with the same node id
                child_id = get_child_id(head_node)
                tail_node.set_node_id(child_id=child_id)
            self.add_node(tail_node)

        if isinstance(head_node, TensorNode):
            self.add_node(head_node)
        self.edge_list.append((tail_node, head_node))

    def add_edge(
        self, tail_id: int, head_id: int, edg_cnt: int
    ) -> None:

        label = None if edg_cnt == 1 else f' x{edg_cnt}'
        self.visual_graph.edge(f'{tail_id}', f'{head_id}', label=label)

    def add_node(self, node: COMPUTATION_NODES) -> None:
        assert node.node_id != 'null'
        if node.node_id not in self.id_dict:
            self.id_dict[node.node_id] = self.running_id
            self.running_id += 1
        label = self.get_node_label(node)
        node_color = ComputationGraph.get_node_color(node)
        self.visual_graph.node(
            name=f'{self.id_dict[node.node_id]}', label=label, fillcolor=node_color
        )

    def get_node_label(self, node: COMPUTATION_NODES) -> str:
        input_str = 'input'
        output_str = 'output'
        if self.show_shapes:
            if isinstance(node, TensorNode):
                label = f"{node.name}-{node.depth}: {node.tensor_shape}"
            else:
                input_repr = compact_list_repr(node.input_shape)
                output_repr = compact_list_repr(node.output_shape)
                label = (
                    f"{node.name}\n-{node.depth}|"
                    f"{{{input_str}:|{output_str}:}}|"
                    f"{{{input_repr}|{output_repr}}}"
                )
        else:
            label = f"{node.name}-{node.depth}"
        return label

    def resize_graph(self, scale=1.0, size_per_element=0.3, min_size=12):
        """Resize the graph according to how much content it contains.
        Modify the graph in place. Default values are subject to change,
        so far they seem to work fine.
        """
        # Get the approximate number of nodes and edges
        num_rows = len(self.visual_graph.body)
        content_size = num_rows * size_per_element
        size = scale * max(min_size, content_size)
        size_str = str(size) + "," + str(size)
        self.visual_graph.graph_attr.update(size=size_str,)

    @staticmethod
    def get_node_color(
        node: COMPUTATION_NODES
    ) -> str:
        return node2color[type(node)]


class RecursionDepth:
    '''Context Manager to increase recursion limit.
    Inside the context, recursion limit is limit (default=2000)
    Outside the context => default limit of python. This is necessary
    for traversal of deep models e.g. resnet151'''
    def __init__(self, limit=2000):
        self.limit = limit
        self.default_limit = sys.getrecursionlimit()

    def __enter__(self):
        sys.setrecursionlimit(self.limit)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        sys.setrecursionlimit(self.default_limit)


def compact_list_repr(x: list):
    '''returns more compact representation of list with
    repeated elements. This is useful for e.g. output of transformer/rnn
    models where hidden state outputs shapes is repetation of one hidden unit
    output'''

    list_counter = Counter(x)
    x_repr = ''

    for elem, cnt in list_counter.items():
        if cnt == 1:
            x_repr += f'{elem}, '
        else:
            x_repr += f'{cnt} x {elem}, '

    # get rid of last comma
    return x_repr[:-2]


def get_child_id(head_node: COMPUTATION_NODES) -> str | int:
    ''' This returns id of child to get correct id.
    This is used to identify the recursively used modules.
    Identification relation is as follows:
        ModuleNodes => by id of nn.Module object
        Parameterless Modules => by id node object
        FunctionNodes => by id of node object
    '''
    if isinstance(head_node, ModuleNode):
        if head_node.is_activation:
            child_id = head_node.node_id
        else:
            child_id = head_node.compute_unit_id
    elif isinstance(head_node, FunctionNode):
        child_id = head_node.node_id
    else:
        child_id = head_node.tensor_id

    return child_id
