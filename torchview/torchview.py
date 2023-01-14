from __future__ import annotations

import warnings
from typing import (
    Sequence, Any, Mapping, Union, Callable, Iterable, Optional,
    Iterator, List
)

import graphviz
import torch
from torch import nn
from torch.jit import ScriptModule

from .computation_node import NodeContainer
from .computation_graph import ComputationGraph
from .computation_node import TensorNode
from .recorder_tensor import (
    module_forward_wrapper, _orig_module_forward, RecorderTensor,
    reduce_data_info, collect_tensor_node, Recorder

)

COMPILED_MODULES = (ScriptModule,)

INPUT_DATA_TYPE = Union[torch.Tensor, Sequence[Any], Mapping[str, Any]]
CORRECTED_INPUT_DATA_TYPE = Optional[Union[Iterable[Any], Mapping[Any, Any]]]
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]
CORRECTED_INPUT_SIZE_TYPE = List[Union[Sequence[Any], torch.Size]]

# TODO: test/change api to include also function calls, not only pytorch models
# so, keep the api here as general as possible


def draw_graph(
    model: nn.Module,
    input_data: INPUT_DATA_TYPE | None = None,
    input_size: INPUT_SIZE_TYPE | None = None,
    graph_name: str = 'model',
    depth: int | float = 3,
    device: torch.device | str | None = None,
    dtypes: list[torch.dtype] | None = None,
    mode: str | None = None,
    strict: bool = True,
    expand_nested: bool = False,
    graph_dir: str | None = None,
    hide_module_functions: bool = True,
    hide_inner_tensors: bool = True,
    roll: bool = False,
    show_shapes: bool = True,
    save_graph: bool = False,
    filename: str | None = None,
    directory: str = '.',
    **kwargs: Any,
) -> ComputationGraph:
    '''Returns visual representation of the input Pytorch Module with
    ComputationGraph object. ComputationGraph object contains:

    1) Root nodes (usually tensor node for input tensors) which connect to all
    the other nodes of computation graph of pytorch module recorded during forward
    propagation.

    2) graphviz.Digraph object that contains visual representation of computation
    graph of pytorch module. This graph visual shows modules/ module hierarchy,
    torch_functions, shapes and tensors recorded during forward prop, for examples
    see documentation, and colab notebooks.


    Args:
        model (nn.Module):
            Pytorch model to represent visually.

        input_data (data structure containing torch.Tensor):
            input for forward method of model. Wrap it in a list for
            multiple args or in a dict or kwargs

        input_size (Sequence of Sizes):
            Shape of input data as a List/Tuple/torch.Size
            (dtypes must match model input, default is FloatTensors).
            Default: None

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

        device (str or torch.device):
            Device to place and input tensors. Defaults to
            gpu if cuda is seen by pytorch, otherwise to cpu.
            Default: None

        dtypes (list of torch.dtype):
            Uses dtypes to set the types of input tensor if
            input size is given.

        mode (str):
            Mode of model to use for forward prop. Defaults
            to Eval mode if not given
            Default: None

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

        save_graph (bool):
            True => Saves output file of graphviz graph
            False => Does not save
            Default: False

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

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode is None:
        model_mode = 'eval'
    else:
        model_mode = mode

    if graph_dir is None:
        graph_dir = 'TB'

    validate_user_params(
        model, input_data, input_size, depth, device, dtypes,
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

    input_recorder_tensor, kwargs_record_tensor, input_nodes = process_input(
        input_data, input_size, kwargs, device, dtypes
    )

    model_graph = ComputationGraph(
        visual_graph, input_nodes, show_shapes, expand_nested,
        hide_inner_tensors, hide_module_functions, roll, depth
    )

    forward_prop(
        model, input_recorder_tensor, device, model_graph,
        model_mode, **kwargs_record_tensor
    )

    model_graph.fill_visual_graph()

    if save_graph:
        model_graph.visual_graph.render(format='png')
    return model_graph


def forward_prop(
    model: nn.Module,
    x: CORRECTED_INPUT_DATA_TYPE,
    device: torch.device | str,
    model_graph: ComputationGraph,
    mode: str,
    **kwargs: Any,
) -> None:
    '''Performs forward propagation of model on RecorderTensor
    inside context to use module_forward_wrapper'''
    saved_model_mode = model.training
    try:
        if mode == 'train':
            model.train()
        elif mode == 'eval':
            model.eval()
        else:
            raise RuntimeError(
                f"Specified model mode not recognized: {mode}"
            )
        new_module_forward = module_forward_wrapper(model_graph)
        with Recorder(_orig_module_forward, new_module_forward, model_graph):
            with torch.no_grad():
                if isinstance(x, (list, tuple)):
                    _ = model.to(device)(*x, **kwargs)
                elif isinstance(x, Mapping):
                    _ = model.to(device)(**x, **kwargs)
                else:
                    # Should not reach this point, since process_input_data ensures
                    # x is either a list, tuple, or Mapping
                    raise ValueError("Unknown input type")
    except Exception as e:
        raise RuntimeError(
            "Failed to run torchgraph see error message"
        ) from e
    finally:
        model.train(saved_model_mode)


def process_input(
    input_data: INPUT_DATA_TYPE | None,
    input_size: INPUT_SIZE_TYPE | None,
    kwargs: Any,
    device: torch.device | str,
    dtypes: list[torch.dtype] | None = None,
) -> tuple[CORRECTED_INPUT_DATA_TYPE, Any, NodeContainer[TensorNode]]:
    """Reads sample input data to get the input size."""
    x = None
    correct_input_size = []
    kwargs_recorder_tensor = traverse_data(kwargs, get_recorder_tensor, type)
    if input_data is not None:
        x = set_device(input_data, device)
        x = traverse_data(x, get_recorder_tensor, type)
        if isinstance(x, RecorderTensor):
            x = [x]

    if input_size is not None:
        if dtypes is None:
            dtypes = [torch.float] * len(input_size)
        correct_input_size = get_correct_input_sizes(input_size)
        x = get_input_tensor(correct_input_size, dtypes, device)

    input_data_node: NodeContainer[TensorNode] = (
        reduce_data_info(
            [x, kwargs_recorder_tensor], collect_tensor_node, NodeContainer()
        )
    )
    return x, kwargs_recorder_tensor, input_data_node


def validate_user_params(
    model: nn.Module,
    input_data: INPUT_DATA_TYPE | None,
    input_size: INPUT_SIZE_TYPE | None,
    depth: int | float,
    device: torch.device | str | None,
    dtypes: list[torch.dtype] | None,
) -> None:
    """Raise exceptions if the user's input is invalid."""
    if depth < 0:
        raise ValueError(
            f"depth must be a non-negative number, depth={depth}"
        )

    if isinstance(model, COMPILED_MODULES):
        warnings.warn(
            "Currently, traced modules are not fully supported. But, there is "
            "a potential solution to support traced models. "
            "For details, see relevant issue in the main repo"
        )

    one_input_specified = (input_data is None) != (input_size is None)
    if not one_input_specified:
        raise RuntimeError("Only one of (input_data, input_size) should be specified.")

    if dtypes is not None and any(
        dtype in (torch.float16, torch.bfloat16) for dtype in dtypes
    ):
        if input_size is not None:
            warnings.warn(
                "Half precision is not supported with input_size parameter, and may "
                "output incorrect results. Try passing input_data directly."
            )

        device_str = device.type if isinstance(device, torch.device) else device
        if device_str == "cpu":
            warnings.warn(
                "Half precision is not supported on cpu. Set the `device` field or "
                "pass `input_data` using the correct device."
            )


def traverse_data(
    data: Any, action_fn: Callable[..., Any], aggregate_fn: Callable[..., Any]
) -> Any:
    """
    Traverses any type of nested data. On a tensor, returns the action given by
    action_fn, and afterwards aggregates the results using aggregate_fn.
    """
    if isinstance(data, torch.Tensor):
        return action_fn(data)

    # Recursively apply to collection items
    aggregate = aggregate_fn(data)
    if isinstance(data, Mapping):
        return aggregate(
            {
                k: traverse_data(v, action_fn, aggregate_fn)
                for k, v in data.items()
            }
        )
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # Named tuple
        return aggregate(
            *(traverse_data(d, action_fn, aggregate_fn) for d in data)
        )
    if isinstance(data, Iterable) and not isinstance(data, str):
        return aggregate(
            [traverse_data(d, action_fn, aggregate_fn) for d in data]
        )
    # Data is neither a tensor nor a collection
    return data


def set_device(data: Any, device: torch.device | str) -> Any:
    """Sets device for all data types and collections of input types."""
    return traverse_data(
        data,
        action_fn=lambda data: data.to(device, non_blocking=True),
        aggregate_fn=type,
    )


def get_recorder_tensor(
    input_tensor: torch.Tensor
) -> RecorderTensor:
    '''returns RecorderTensor version of input_tensor with
    TensorNode instance attached to it'''

    # as_subclass is necessary for torch versions < 3.12
    input_recorder_tensor: RecorderTensor = input_tensor.as_subclass(RecorderTensor)
    input_recorder_tensor.tensor_nodes = []
    input_node = TensorNode(
        tensor=input_recorder_tensor,
        depth=0,
        name='input-tensor',
    )

    input_recorder_tensor.tensor_nodes.append(input_node)
    return input_recorder_tensor


def get_input_tensor(
    input_size: CORRECTED_INPUT_SIZE_TYPE,
    dtypes: list[torch.dtype],
    device: torch.device | str,
) -> list[RecorderTensor]:
    """Get input_tensor for use in model.forward()"""
    x = []
    for size, dtype in zip(input_size, dtypes):
        input_tensor = torch.rand(*size)
        x.append(
            get_recorder_tensor(input_tensor.to(device).type(dtype))
        )
    return x


def get_correct_input_sizes(input_size: INPUT_SIZE_TYPE) -> CORRECTED_INPUT_SIZE_TYPE:
    """
    Convert input_size to the correct form, which is a list of tuples.
    Also handles multiple inputs to the network.
    """
    if not isinstance(input_size, (list, tuple)):
        raise TypeError(
            "Input_size is not a recognized type. Please ensure input_size is valid.\n"
            "For multiple inputs to the network, ensure input_size is a list of tuple "
            "sizes. If you are having trouble here, please submit a GitHub issue."
        )
    if not input_size or any(size <= 0 for size in flatten(input_size)):
        raise ValueError("Input_data is invalid, or negative size found in input_data.")

    if isinstance(input_size, list) and isinstance(input_size[0], int):
        return [tuple(input_size)]
    if isinstance(input_size, list):
        return input_size
    if isinstance(input_size, tuple) and isinstance(input_size[0], tuple):
        return list(input_size)
    return [input_size]


def flatten(nested_array: INPUT_SIZE_TYPE) -> Iterator[Any]:
    """Flattens a nested array."""
    for item in nested_array:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item
