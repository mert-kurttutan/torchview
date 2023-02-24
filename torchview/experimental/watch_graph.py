from __future__ import annotations

import graphviz

from ..computation_graph import ComputationGraph
from ..recorder_tensor import (
    module_forward_wrapper, _orig_module_forward, Recorder

)

# TODO: test/change api to include also function calls, not only pytorch models
# so, keep the api here as general as possible


def watch_graph(
    graph_name: str = 'model',
    depth: int | float = 3,
    strict: bool = True,
    expand_nested: bool = False,
    graph_dir: str | None = None,
    hide_module_functions: bool = True,
    hide_inner_tensors: bool = True,
    roll: bool = False,
    show_shapes: bool = True,
    filename: str | None = None,
    directory: str = '.',
) -> Recorder:
    '''
    This is Experimental Do not Rely on this. Further details will be explained
    in the docs. Returns context in which to watch graph execution, no complete yet

    1) Root nodes (usually tensor node for input tensors) which connect to all
    the other nodes of computation graph of pytorch module recorded during forward
    propagation.

    2) graphviz.Digraph object that contains visual representation of computation
    graph of pytorch module. This graph visual shows modules/ module hierarchy,
    torch_functions, shapes and tensors recorded during forward prop, for examples
    see documentation, and colab notebooks.


    Args:

        graph_name (str):
            Name for graphviz.Digraph object. Also default name graphviz file
            of Graph Visualization
            Default: 'model'

        depth (int):
            Upper limit for depth of nodes to be shown in visualization.
            Depth is measured how far is module/tensor inside the module hierarchy.
            For instance, main module has depth=0, whereas submodule of main module
            has depth=1, and so on.
            Default: 3

        strict (bool):
            if true, graphviz visual does not allow multiple edges
            between nodes. Mutiple edge occurs e.g. when there are tensors
            from module node to module node and hiding those tensors
            Default: True

        expand_nested (bool):
            if true, shows nested modules with dashed borders

        graph_dir (str):
            Sets the direction of visual graph
            'TB' -> Top to Bottom
            'LR' -> Left to Right
            'BT' -> Bottom to Top
            'RL' -> Right to Left
            Default: None -> TB

        hide_module_function (bool):
            Determines whether to hide module torch_functions. Some
            modules consist only of torch_functions (no submodule),
            e.g. nn.Conv2d.
            True => Dont include module functions in graphviz
            False => Include modules function in graphviz
            Default: True

        hide_inner_tensors (bool):
            Inner tensor is all the tensors of computation graph
            but input and output tensors
            True => Does not show inner tensors in graphviz
            False => Shows inner tensors in graphviz
            Default: True

        roll (bool):
            If true, rolls recursive modules.
            Default: False

        show_shapes (bool):
            True => Show shape of tensor, input, and output
            False => Dont show
            Default: True

        filename (str):
            name of the file to store dot syntax representation and
            image file of graphviz graph. Defaults to graph_name

        directory (str):
            directory in which to store graphviz output files.
            Default: .

    Returns:
        ComputationGraph object that contains visualization of the input
        pytorch model in the form of graphviz Digraph object
    '''

    if filename is None:
        filename = f'{graph_name}.gv'

    if graph_dir is None:
        graph_dir = 'TB'

    validate_user_params(
        depth,
    )

    graph_attr = {
        'ordering': 'in',
        'rankdir': graph_dir,
    }

    # visual settings from torchviz
    # seems to work visually well
    node_attr = {
        'style': 'filled',
        'shape': 'plaintext',
        'align': 'left',
        'fontsize': '10',
        'ranksep': '0.1',
        'height': '0.2',
        'fontname': 'Linux libertine',
        'margin': '0',
    }

    edge_attr = {
        'fontsize': '10',
    }
    visual_graph = graphviz.Digraph(
        name=graph_name, engine='dot', strict=strict,
        graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr,
        directory=directory, filename=filename
    )

    input_nodes = 0
    model_graph = ComputationGraph(
        visual_graph, input_nodes, show_shapes, expand_nested,   # type: ignore[arg-type]
        hide_inner_tensors, hide_module_functions, roll, depth
    )

    new_module_forward = module_forward_wrapper(model_graph)

    return Recorder(_orig_module_forward, new_module_forward, model_graph)


def validate_user_params(
    depth: int | float,
) -> None:
    """Raise exceptions if the user's input is invalid."""
    if depth < 0:
        raise ValueError(
            f"depth must be a non-negative number, depth={depth}"
        )
