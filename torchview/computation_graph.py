# mypy: ignore-errors
from __future__ import annotations

from typing import Union
from collections import Counter
from contextlib import nullcontext


from graphviz import Digraph
from torch.nn.modules import Identity

from .computation_node import NodeContainer
from .computation_node import TensorNode, ModuleNode, FunctionNode
from .utils import updated_dict

COMPUTATION_NODES = Union[TensorNode, ModuleNode, FunctionNode]

# TODO: Currently, we only use directed graphviz graph since DNN are
# graphs except for e.g. graph neural network (GNN). Experiment on GNN
# and see if undirected graphviz graph can be used to represent GNNs

# TODO: change api of to include also function calls, not only pytorch models
# so, keep the api here as general as possible

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
        show_shapes: bool = True,
        expand_nested: bool = False,
        hide_inner_tensors: bool = True,
        hide_module_functions: bool = True,
        roll: bool = True,
        depth: int | float = 3,
    ):
        '''
        Resets the running_node_id, id_dict when a new ComputationGraph is initialized.
        Otherwise, labels would depend on previous ComputationGraph runs
        '''
        self.visual_graph = visual_graph
        self.root_container = root_container
        self.show_shapes = show_shapes
        self.expand_nested = expand_nested
        self.hide_inner_tensors = hide_inner_tensors
        self.hide_module_functions = hide_module_functions
        self.roll = roll
        self.depth = depth

        self.reset_graph_history()

    def reset_graph_history(self):
        '''Resets to id config to the setting of empty visual graph
        needed for getting reproducible/deterministic node name and
        graphviz graphs. This is especially important for output tests
        '''
        self.running_node_id: int = 0
        self.running_subgraph_id: int = 0
        self.id_dict: dict[str, int] = {}

        self.edge_list: list[tuple[COMPUTATION_NODES, COMPUTATION_NODES]] = []

        # module node  to capture whole graph
        main_container_module = ModuleNode(Identity(), -1)
        self.subgraph_dict: dict[str, int] = {main_container_module.node_id: 0}
        self.running_subgraph_id += 1

        # Add input nodes
        self.node_hierarchy = {
            main_container_module: list(root_node for root_node in self.root_container)
        }
        for root_node in self.root_container:
            root_node.context = self.node_hierarchy[main_container_module]

    def fill_visual_graph(self):
        '''Fills the graphviz graph with desired nodes and edges.'''

        # First add input nodes
        for root_node in self.root_container:
            root_node.name = 'input-tensor'
            self.add_node(root_node)

        self.render_nodes()
        self.render_edges()
        self.resize_graph()

    def render_nodes(
        self,
    ):
        kwargs = {
            'cur_node': self.node_hierarchy,
            'subgraph': None,
        }
        self.traverse_graph(self.collect_graph, **kwargs)

    def render_edges(self):
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
        self, action_fn, **kwargs
    ):
        cur_node = kwargs['cur_node']
        cur_subgraph = (
            self.visual_graph if kwargs['subgraph'] is None else kwargs['subgraph']
        )
        if isinstance(cur_node, (TensorNode, ModuleNode, FunctionNode)):
            if cur_node.depth <= self.depth:
                action_fn(**kwargs)

        elif isinstance(cur_node, dict):
            k, v = list(cur_node.items())[0]
            new_kwargs = updated_dict(kwargs, 'cur_node', k)
            if k.depth <= self.depth and k.depth >= 0:
                action_fn(**new_kwargs)

            display_nested = (
                k.depth < self.depth and k.depth >= 1 and self.expand_nested
            )
            with (
                cur_subgraph.subgraph(name=f'cluster_{self.subgraph_dict[k.node_id]}')
                if display_nested else nullcontext()
            ) as cur_cont:
                if display_nested:
                    cur_cont.attr(
                        style='dashed', label=k.name, labeljust='l', fontsize='12'
                    )
                    new_kwargs = updated_dict(new_kwargs, 'subgraph', cur_cont)
                for g in v:
                    new_kwargs = updated_dict(new_kwargs, 'cur_node', g)
                    self.traverse_graph(action_fn, **new_kwargs)
        else:
            raise ValueError('this should not be reached')

    def collect_graph(
        self, **kwargs,
    ) -> None:
        '''Adds edges and nodes with appropriate node name/id (so it respects
        properties e.g. if rolled recursive nodes are given the same node name
        in graphviz graph)'''

        cur_node = kwargs['cur_node']
        self.check_node(cur_node)
        is_visible = self.is_node_visible(cur_node)
        # add node
        if is_visible:
            subgraph = kwargs['subgraph']
            if isinstance(cur_node, (FunctionNode, ModuleNode)):
                if self.roll:
                    self.rollify(cur_node)
                self.add_node(cur_node, subgraph)

            if isinstance(cur_node, TensorNode):
                self.add_node(cur_node, subgraph)

        elif isinstance(cur_node, ModuleNode):
            # add subgraph
            if self.roll:
                self.rollify(cur_node)
            if cur_node.node_id not in self.subgraph_dict:
                self.subgraph_dict[cur_node.node_id] = self.running_subgraph_id
                self.running_subgraph_id += 1

        if not isinstance(cur_node, TensorNode):
            return

        # add edges
        # {cur_node -> head} part
        tail_node = self.get_tail_node(cur_node)
        if cur_node.outputs:
            for output_node in cur_node.outputs:
                assert not isinstance(output_node, TensorNode)
                is_visible = self.is_node_visible(output_node)
                if is_visible:
                    if self.hide_inner_tensors:
                        self.edge_list.append((tail_node, output_node))
                    else:
                        self.edge_list.append((cur_node, output_node))

        # {tail -> cur_node} part
        # # output node
        is_tensor_visible = self.is_node_visible(cur_node)
        # visible tensor and non-input tensor nodes
        if is_tensor_visible and not isinstance(tail_node, TensorNode):
            tail_node = self.get_tail_node(cur_node)
            self.edge_list.append((tail_node, cur_node))

    def rollify(
        self, cur_node: COMPUTATION_NODES
    ):
        head_node = next(iter(cur_node.end_nodes))
        if head_node.outputs and self.hide_inner_tensors:
            head_node = next(iter(head_node.outputs))

        # identify recursively used modules
        # with the same node id
        output_id = get_output_id(head_node)
        cur_node.set_node_id(output_id=output_id)

    def is_node_visible(self, compute_node):
        if isinstance(compute_node, (ModuleNode, FunctionNode)):
            is_visible = (
                isinstance(compute_node, FunctionNode) or (
                    (self.hide_module_functions and compute_node.is_container)
                    or compute_node.depth == self.depth
                )
            )
            return is_visible

        if isinstance(compute_node, TensorNode):
            is_visible = (
                not self.hide_inner_tensors or
                (not compute_node.inputs or not compute_node.outputs)
            )

            if compute_node.is_aux:
                input_node = next(iter(compute_node.inputs))
                is_input_visible = self.is_node_visible(input_node)
                is_visible = is_visible and is_input_visible
            return is_visible

        raise ValueError(
            'Only Computation Nodes are allowed for this function as an input'
        )

    def get_tail_node(self, _tensor_node):
        tensor_node = _tensor_node
        # non-output nodes eminating from input node
        if not tensor_node.main_node.inputs and tensor_node.outputs:
            return tensor_node.main_node

        # inner auxiliary nodes
        if tensor_node.outputs and tensor_node.is_aux:
            # empty module should display auxiliary nodes
            if next(iter(tensor_node.inputs)).depth < tensor_node.depth:
                tensor_node = tensor_node.main_node

        sorted_depth = sorted(depth for depth in tensor_node.input_hierarchy)
        chosen_input = next(iter(tensor_node.inputs))
        for depth in sorted_depth:
            chosen_input = tensor_node.input_hierarchy[depth]
            if depth == self.depth:
                break

        return chosen_input

    def add_edge(
        self, tail_id: int, head_id: int, edg_cnt: int
    ) -> None:

        label = None if edg_cnt == 1 else f' x{edg_cnt}'
        self.visual_graph.edge(f'{tail_id}', f'{head_id}', label=label)

    def add_node(
        self, node: COMPUTATION_NODES, subgraph: Digraph | None = None
    ) -> None:
        assert node.node_id != 'null', f'wrong id {node} {type(node)}'
        if node.node_id not in self.id_dict:
            self.id_dict[node.node_id] = self.running_node_id
            self.running_node_id += 1
        label = self.get_node_label(node)
        node_color = ComputationGraph.get_node_color(node)

        if subgraph is None:
            subgraph = self.visual_graph
        subgraph.node(
            name=f'{self.id_dict[node.node_id]}', label=label, fillcolor=node_color,
        )

    def get_node_label(self, node: COMPUTATION_NODES) -> str:
        input_str = 'input'
        output_str = 'output'
        new_line = '&#92;n'  # needed for record shape config in graphviz
        if self.show_shapes:
            if isinstance(node, TensorNode):
                label = (
                    f"{node.name}{new_line}depth:{node.depth}|"
                    f"{node.tensor_shape}"
                )
            else:
                input_repr = compact_list_repr(node.input_shape)
                output_repr = compact_list_repr(node.output_shape)
                label = (
                    f"{node.name}{new_line}depth:{node.depth}|"
                    f"{{{input_str}:|{output_str}:}}|"
                    f"{{{input_repr}|{output_repr}}}"
                )
        else:
            label = f"{node.name}{new_line}depth:{node.depth}"
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

    def check_node(self, node):
        assert '-' not in node.node_id, 'No repetition of node recording'
        assert any(node.outputs) or any(node.inputs), f'isolated node! {node}'
        assert node.depth <= self.depth, f"exceeds depth limit, {node}"
        assert (
            sum(1 for _ in node.inputs) in [0, 1] or not isinstance(node, TensorNode)
        ), (
            f'tensor must have single input node {node}'
        )


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


def get_output_id(head_node: COMPUTATION_NODES) -> str | int:
    ''' This returns id of output to get correct id.
    This is used to identify the recursively used modules.
    Identification relation is as follows:
        ModuleNodes => by id of nn.Module object
        Parameterless Modules => by id node object
        FunctionNodes => by id of node object
    '''
    if isinstance(head_node, ModuleNode):
        if head_node.is_activation:
            output_id = head_node.node_id
        else:
            output_id = head_node.compute_unit_id
    elif isinstance(head_node, FunctionNode):
        output_id = head_node.node_id
    else:
        output_id = head_node.node_id

    return output_id
