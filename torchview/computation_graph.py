# mypy: ignore-errors
from __future__ import annotations

from typing import Union, Any, Callable
from collections import Counter
from contextlib import nullcontext

from graphviz import Digraph
from torch.nn.modules import Identity

from .computation_node import NodeContainer
from .computation_node import TensorNode, ModuleNode, FunctionNode
from .utils import updated_dict, assert_input_type

COMPUTATION_NODES = Union[TensorNode, ModuleNode, FunctionNode]

node2color = {
    TensorNode: "lightyellow",
    ModuleNode: "darkseagreen1",
    FunctionNode: "aliceblue",
}

# TODO: Currently, we only use directed graphviz graph since DNN are
# graphs except for e.g. graph neural network (GNN). Experiment on GNN
# and see if undirected graphviz graph can be used to represent GNNs

# TODO: change api of to include also function calls, not only pytorch models
# so, keep the api here as general as possible


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

        hide_module_functions (bool):
            Some modules contain only torch.function and no submodule,
            e.g. nn.Conv2d. They are usually implemented to do one type
            of computation, e.g. Conv2d -> 2D Convolution. If True,
            visual graph only displays the module itself,
            while ignoring its inner functions.

        hide_inner_tensors (bool):
            Whether to hide inner tensors in graphviz graph object


        node_hierarchy dict:
            Represents nested hierarchy of ComputationNodes by nested dictionary
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

        # specs for html table, needed for node labels
        self.html_config = {
            'border': 0,
            'cell_border': 1,
            'cell_spacing': 0,
            'cell_padding': 4,
            'col_span': 2,
            'row_span': 2,
        }
        self.reset_graph_history()

    def reset_graph_history(self) -> None:
        '''Resets to id config to the setting of empty visual graph
        needed for getting reproducible/deterministic node name and
        graphviz graphs. This is especially important for output tests
        '''
        self.context_tracker = {'current_context': [], 'current_depth': 0}
        self.running_node_id: int = 0
        self.running_subgraph_id: int = 0
        self.id_dict: dict[str, int] = {}
        self.node_set: set[int] = set()
        self.edge_list: list[tuple[COMPUTATION_NODES, COMPUTATION_NODES]] = []

        # module node  to capture whole graph
        main_container_module = ModuleNode(Identity(), -1)
        main_container_module.is_container = False
        self.subgraph_dict: dict[str, int] = {main_container_module.node_id: 0}
        self.running_subgraph_id += 1

        # Add input nodes
        self.node_hierarchy = {
            main_container_module: list(self.root_container)
        }
        for root_node in self.root_container:
            root_node.context = self.node_hierarchy[main_container_module]

    def fill_visual_graph(self) -> None:
        '''Fills the graphviz graph with desired nodes and edges.'''

        self.render_nodes()
        self.render_edges()
        self.resize_graph()

    def render_nodes(self) -> None:
        kwargs = {
            'cur_node': self.node_hierarchy,
            'subgraph': None,
        }
        self.traverse_graph(self.collect_graph, **kwargs)

    def render_edges(self) -> None:
        '''Records all edges in self.edge_list to
        the graphviz graph using node ids from edge_list'''
        edge_counter: dict[tuple[int, int], int] = {}
        for tail, head in self.edge_list:
            edge_id = self.id_dict[tail.node_id], self.id_dict[head.node_id]
            edge_counter[edge_id] = edge_counter.get(edge_id, 0) + 1
            self.add_edge(edge_id, edge_counter[edge_id])

    def traverse_graph(
        self, action_fn: Callable[..., None], **kwargs: Any
    ) -> None:
        cur_node = kwargs['cur_node']
        cur_subgraph = (
            self.visual_graph if kwargs['subgraph'] is None else kwargs['subgraph']
        )
        assert_input_type(
            'traverse_graph', (TensorNode, ModuleNode, FunctionNode, dict), cur_node
        )
        if isinstance(cur_node, (TensorNode, ModuleNode, FunctionNode)):
            if cur_node.depth <= self.depth:
                action_fn(**kwargs)
            return

        if isinstance(cur_node, dict):
            k, v = list(cur_node.items())[0]
            new_kwargs = updated_dict(kwargs, 'cur_node', k)
            if k.depth <= self.depth and k.depth >= 0:
                action_fn(**new_kwargs)

            # if it is container module, move directly to outputs
            if self.hide_module_functions and k.is_container:
                for g in k.output_nodes:
                    new_kwargs = updated_dict(new_kwargs, 'cur_node', g)
                    self.traverse_graph(action_fn, **new_kwargs)
                return

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

    def collect_graph(self, **kwargs: Any) -> None:
        '''Adds edges and nodes with appropriate node name/id (so it respects
        properties e.g. if rolled recursive nodes are given the same node name
        in graphviz graph)'''

        cur_node = kwargs['cur_node']
        # if tensor node is traced, dont repeat collecting
        # if node is isolated, dont record it
        is_isolated = cur_node.is_root() and cur_node.is_leaf()
        if id(cur_node) in self.node_set or is_isolated:
            return

        self.check_node(cur_node)
        is_cur_visible = self.is_node_visible(cur_node)
        # add node
        if is_cur_visible:
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

        # add edges only through
        # node -> TensorNode -> Node connection
        if not isinstance(cur_node, TensorNode):
            return

        # add edges
        # {cur_node -> head} part
        tail_node = self.get_tail_node(cur_node)
        is_main_node_visible = self.is_node_visible(cur_node.main_node)
        is_tail_node_visible = self.is_node_visible(tail_node)
        if not cur_node.is_leaf():
            for children_node in cur_node.children:
                is_output_visible = self.is_node_visible(children_node)
                if is_output_visible:
                    if is_main_node_visible:
                        self.edge_list.append((cur_node, children_node))
                    elif is_tail_node_visible:
                        self.edge_list.append((tail_node, children_node))

        # {tail -> cur_node} part
        # # output node
        # visible tensor and non-input tensor nodes
        if is_cur_visible and not cur_node.is_root():
            assert not isinstance(tail_node, TensorNode) or tail_node.is_root(), (
                "get_tail_node function returned inconsistent Node, please report this"
            )
            self.edge_list.append((tail_node, cur_node))

    def rollify(self, cur_node: ModuleNode | FunctionNode) -> None:
        '''Rolls computational graph by identifying recursively used
        Modules. This is done by giving the same id for nodes that are
        recursively used.
        This becomes complex when there are stateless and torch.functions.
        For more details see docs'''

        head_node = next(iter(cur_node.output_nodes))
        if not head_node.is_leaf() and self.hide_inner_tensors:
            head_node = next(iter(head_node.children))

        # identify recursively used modules
        # with the same node id
        output_id = get_output_id(head_node)
        cur_node.set_node_id(output_id=output_id)

    def is_node_visible(self, compute_node: COMPUTATION_NODES) -> bool:
        '''Returns True if node should be displayed on the visual
        graph. Otherwise False'''

        assert_input_type(
            'is_node_visible', (TensorNode, ModuleNode, FunctionNode,), compute_node
        )

        if compute_node.name == 'empty-pass':
            return False

        if isinstance(compute_node, (ModuleNode, FunctionNode)):
            is_visible = (
                isinstance(compute_node, FunctionNode) or (
                    (self.hide_module_functions and compute_node.is_container)
                    or compute_node.depth == self.depth
                )
            )
            return is_visible

        else:
            if compute_node.main_node.depth < 0 or compute_node.is_aux:
                return False

            is_main_input_or_output = (
                (compute_node.is_root() or compute_node.is_leaf())
                and compute_node.depth == 0
            )
            is_visible = (
                not self.hide_inner_tensors or is_main_input_or_output
            )

            return is_visible

    def get_tail_node(self, _tensor_node: TensorNode) -> COMPUTATION_NODES:

        tensor_node = _tensor_node.main_node if _tensor_node.is_aux else _tensor_node

        # non-output nodes eminating from input node
        if tensor_node.is_root():
            return tensor_node

        current_parent_h = tensor_node.parent_hierarchy

        sorted_depth = sorted(depth for depth in current_parent_h)
        tail_node = next(iter(tensor_node.parents))
        depth = 0
        for depth in sorted_depth:
            tail_node = current_parent_h[depth]
            if depth >= self.depth:
                break

        module_depth = depth - 1
        # if returned by container module and hide_module_functions
        if (
            isinstance(current_parent_h[depth], FunctionNode) and
            module_depth in tensor_node.parent_hierarchy and self.hide_module_functions
        ):
            if current_parent_h[module_depth].is_container:
                return current_parent_h[module_depth]

        # Even though this is recursive, not harmful for complexity
        # The reason: the (time) complexity ~ O(L^2) where L
        # is the length of CONTINUOUS path along which the same tensor is passed
        # without any operation on it. L is always small since we dont use
        # infinitely big network with infinitely big continuou pass of unchanged
        # tensor. This recursion is necessary e.g. for LDC model
        if tail_node.name == 'empty-pass':
            empty_pass_parent = next(iter((tail_node.parents)))
            assert isinstance(empty_pass_parent, TensorNode), (
                f'{empty_pass_parent} is input of {tail_node}'
                f'and must a be TensorNode'
            )
            return self.get_tail_node(empty_pass_parent)
        return tail_node

    def add_edge(
        self, edge_ids: tuple[int, int], edg_cnt: int
    ) -> None:

        tail_id, head_id = edge_ids
        label = None if edg_cnt == 1 else f' x{edg_cnt}'
        self.visual_graph.edge(f'{tail_id}', f'{head_id}', label=label)

    def add_node(
        self, node: COMPUTATION_NODES, subgraph: Digraph | None = None
    ) -> None:
        '''Adds node to the graphviz with correct id, label and color
        settings. Updates state of running_node_id if node is not
        identified before.'''
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
        self.node_set.add(id(node))

    def get_node_label(self, node: COMPUTATION_NODES) -> str:
        '''Returns html-like format for the label of node. This html-like
        label is based on Graphviz API for html-like format. For setting of node label
        it uses graph config and html_config.'''
        input_str = 'input'
        output_str = 'output'
        border = self.html_config['border']
        cell_sp = self.html_config['cell_spacing']
        cell_pad = self.html_config['cell_padding']
        cell_bor = self.html_config['cell_border']
        if self.show_shapes:
            if isinstance(node, TensorNode):
                label = f'''<
                    <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
                    CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
                        <TR><TD>{node.name}<BR/>depth:{node.depth}</TD><TD>{node.tensor_shape}</TD></TR>
                    </TABLE>>'''
            else:
                input_repr = compact_list_repr(node.input_shape)
                output_repr = compact_list_repr(node.output_shape)
                label = f'''<
                    <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
                    CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
                    <TR>
                        <TD ROWSPAN="2">{node.name}<BR/>depth:{node.depth}</TD>
                        <TD COLSPAN="2">{input_str}:</TD>
                        <TD COLSPAN="2">{input_repr} </TD>
                    </TR>
                    <TR>
                        <TD COLSPAN="2">{output_str}: </TD>
                        <TD COLSPAN="2">{output_repr} </TD>
                    </TR>
                    </TABLE>>'''
        else:
            label = f'''<
                    <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
                    CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
                        <TR><TD>{node.name}<BR/>depth:{node.depth}</TD></TR>
                    </TABLE>>'''
        return label

    def resize_graph(
        self,
        scale: float = 1.0,
        size_per_element: float = 0.3,
        min_size: float = 12
    ) -> None:
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

    def check_node(self, node: COMPUTATION_NODES) -> None:
        assert node.node_id != 'null', f'wrong id {node} {type(node)}'
        assert '-' not in node.node_id, 'No repetition of node recording is allowed'
        assert node.depth <= self.depth, f"Exceeds display depth limit, {node}"
        assert (
            sum(1 for _ in node.parents) in [0, 1] or not isinstance(node, TensorNode)
        ), (
            f'tensor must have single input node {node}'
        )


def compact_list_repr(x: list[Any]) -> str:
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


def get_output_id(head_node: COMPUTATION_NODES) -> str:
    ''' This returns id of output to get correct id.
    This is used to identify the recursively used modules.
    Identification relation is as follows:
        ModuleNodes => by id of nn.Module object
        Parameterless ModulesNodes => by id of nn.Module object
        FunctionNodes => by id of Node object
    '''
    if isinstance(head_node, ModuleNode):
        output_id = str(head_node.compute_unit_id)
    else:
        output_id = head_node.node_id

    return output_id
