from __future__ import annotations

from typing import Any, Iterable, Mapping, TypeVar
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch._C import ScriptMethod

from .computation_node import ModuleNode, FunctionNode, TensorNode, NodeContainer
from .computation_graph import ComputationGraph

from .utils import OrderedSet

# Needed for module wrapper and resetting
_orig_module_forward = torch.nn.Module.__call__


# Functions below are torch.tensor creation operations
# Inside Recorder context, we wrapped these functions
# so that these return RecorderTensor can be recorded during
# forward propagation
orig_name_list = [
    "as_tensor", "from_numpy", "zeros", "zeros_like",
    "ones", "ones_like", "arange", "range", "linspace",
    "logspace", "eye", "empty", "empty_like", "full",
    "full_like", "complex", "heaviside", "bernoulli",
    "multinomial", "normal", "poisson", "rand", "rand_like",
    "randint", "randint_like", "randn", "randn_like",
    "randperm"
]
_orig_op_list = [getattr(torch, name) for name in orig_name_list]


class Recorder:
    '''Context Manager that sets modules forward and torch creation ops
    to record them in computation graph'''
    def __init__(
        self, orig_mod_forward: Callable[..., Any], new_mod_forward: Callable[..., Any],
        model_graph: ComputationGraph
    ) -> None:
        self.orig_module_forward = orig_mod_forward
        self.new_module_forward = new_mod_forward
        self.model_graph = model_graph

    def __enter__(self) -> None:
        setattr(
            torch.nn.Module, "__call__", self.new_module_forward
        )

        for name, op in zip(orig_name_list, _orig_op_list):
            setattr(
                torch, name, creation_ops_wrapper(op, self.model_graph)
            )

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        # reset module __call__ back to original method and torch creation ops
        setattr(
            torch.nn.Module, "__call__", self.orig_module_forward
        )

        for name, op in zip(orig_name_list, _orig_op_list):
            setattr(
                torch, name, op
            )


def creation_ops_wrapper(
    _orig_op: Callable[..., Any], model_graph: ComputationGraph
) -> Callable[..., Any]:
    def _func(*args: Any, **kwargs: Any) -> RecorderTensor:

        input_tensor = _orig_op(*args, **kwargs)
        current_depth = model_graph.context_tracker['current_depth']
        current_context = model_graph.context_tracker['current_context']

        input_recorder_tensor: RecorderTensor = input_tensor.as_subclass(RecorderTensor)
        input_node = TensorNode(
            tensor=input_recorder_tensor,
            depth=current_depth,  # type: ignore[arg-type]
            name='input-tensor' if current_depth == 0 else 'hidden-tensor',
            context=current_context
        )
        current_context.append(input_node)  # type: ignore[attr-defined]
        input_recorder_tensor.tensor_nodes = [input_node]

        return input_recorder_tensor
    return _func


def module_forward_wrapper(model_graph: ComputationGraph) -> Callable[..., Any]:
    '''Wrapper for forward functions of modules'''
    def _module_forward_wrapper(mod: nn.Module, *args: Any, **kwargs: Any) -> Any:
        '''Forward prop of module for RecorderTensor subclass
        Construct Module Node => forward-prop => process output nodes to retain
        module hierarchy correctly
        '''
        # Create module node and connect to its parents tensor node
        input_nodes: NodeContainer[TensorNode] = (
            reduce_data_info([args, kwargs], collect_tensor_node, NodeContainer())
        )
        # get unique input tensors, prevent duplications for auxiliary nodes
        input_recorder: OrderedSet[RecorderTensor] = (
            reduce_data_info([args, kwargs], collect_tensor, OrderedSet())
        )
        # if none of args originated from input
        # hence only torch.Tensor
        if not input_nodes:
            return _orig_module_forward(mod, *args, **kwargs)

        # Create module_node and connect to its parents tensor node
        cur_depth = next(iter(input_nodes)).depth
        input_context = next(iter(input_nodes)).context
        cur_node = ModuleNode(
            mod, cur_depth, input_nodes,  # type: ignore[arg-type]
            name=type(mod).__name__
        )
        cur_node.set_input_shape(
            reduce_data_info([args, kwargs], collect_shape, [])
        )

        # update context with current modules's context
        input_context.append({cur_node: []})
        for node in input_nodes:
            node.add_child(cur_node)

        tensor_to_node: dict[RecorderTensor, TensorNode] = (
            reduce_data_info([args, kwargs], collect_tensor_node_id_dict, {})
        )
        attach_kwargs = {
            'parents': cur_node, 'depth': cur_depth+1,
            'context': input_context[-1][cur_node], 'is_aux': True,
            'name': 'auxiliary-tensor'
        }

        traverse_data_inplace(
            input_recorder, attach_node(attach_kwargs, tensor_to_node)
        )

        model_graph.context_tracker['current_depth'] = cur_depth+1
        model_graph.context_tracker['current_context'] = input_context[-1][cur_node]

        # TODO: check if output contains RecorderTensor
        # this seems not to be necessary so far
        out = _orig_module_forward(mod, *args, **kwargs)

        model_graph.context_tracker['current_depth'] = cur_depth
        model_graph.context_tracker['current_context'] = input_context

        # pop appropriate nodes, see implementation below
        output_recorder: OrderedSet[RecorderTensor] = (
            reduce_data_info(out, collect_tensor, OrderedSet())
        )

        traverse_data_inplace(
            output_recorder,
            process_output_node(cur_node)
        )

        traverse_data_inplace(
            input_recorder, pop_after_forward, recorded_output=output_recorder,
        )

        # remove auxiliary tensor nodes from recorder_tensor
        output_nodes: NodeContainer[TensorNode] = (
            reduce_data_info(out, collect_tensor_node, NodeContainer())
        )

        for output_node in output_nodes:
            cur_node.add_output_nodes(output_node)
            output_node.context = input_context

        cur_node.set_output_shape(reduce_data_info(out, collect_shape, []))
        return out

    return _module_forward_wrapper


class RecorderTensor(torch.Tensor):
    '''Subclass of torch.Tensor used for constructing visual computation graph.

    This class stores list of TensorNode objects to keep record of Nodes during forward
    propagation. The torch_function is also overriden to record needed nodes for visual
    computation graph.

    Attributes:
        All the inherited attributes from torch.Tensor
        tensor_nodes: list[TensorNode]
            List of TensorNode objects to store relevant TensorNodes'''
    @staticmethod
    def __new__(
        cls: Any,
        x: Any,
        tensor_nodes: Any,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        # pylint: disable=unused-argument
        return super().__new__(cls, x, *args, **kwargs)  # type: ignore[call-arg]

    def __init__(
        self, x: Any, tensor_node: TensorNode | list[TensorNode]
    ):
        # pylint: disable=unused-argument
        # super().__init__() # optional

        if isinstance(tensor_node, TensorNode):
            self.tensor_nodes = [tensor_node]
        else:
            self.tensor_nodes = tensor_node

    @classmethod
    def __torch_function__(
        cls: Any, func: Callable[..., Any] | ScriptMethod,
        types: Any,
        args: Any = (),
        kwargs: Any = None,
    ) -> Any:
        '''Calls torch functions for RecorderTensor subclass of torch.Tensor
        Forward prop => Construct Function Node => Construct Output TensorNode
        Args:
            The same arguments as that of  original __torch_function__
            except that the tensor that originated from input (through forward prop)
            are RecorderTensors
        '''
        if kwargs is None:
            kwargs = {}

        args_nodes: NodeContainer[TensorNode] = (
            reduce_data_info([args, kwargs], collect_tensor_node, NodeContainer())
        )

        # This is necessary for torch version < 1.10
        if func in [F.linear, F.embedding]:
            out = nn.parameter.Parameter.__torch_function__(
                func, types, args, kwargs).as_subclass(RecorderTensor)
        else:
            # use original torch_function; otherwise,
            # it leads to infinite recursive call of torch_function
            out = super().__torch_function__(func, types, args, kwargs)

        # if no RecorderTensor is found in input or output
        # dont create any node, give the result only
        if not args_nodes:
            return out
        if not reduce_data_info(out, collect_tensor, OrderedSet()):
            return out

        # Create function_node and connect to its parents tensor node
        cur_depth = next(iter(args_nodes)).depth
        input_context = next(iter(args_nodes)).context
        func_name = (
            func.name if isinstance(func, ScriptMethod) else func.__name__
        )
        cur_node = FunctionNode(
            func, cur_depth, args_nodes, name=func_name  # type: ignore[arg-type]
        )

        for i in args_nodes:
            i.add_child(cur_node)

        input_context.append(cur_node)
        attach_kwargs = {
            'parents': cur_node, 'depth': cur_depth, "context": input_context,
            'is_aux': False, 'parent_hierarchy': {cur_depth: cur_node},
            'name': 'output-tensor' if cur_depth == 0 else 'hidden-tensor'
        }
        traverse_data_inplace(out, attach_node(attach_kwargs))

        # note that when processing inplace operation, input shape is calculated
        # correctly only if inplace operation preserves the input shape
        # which it does for all torch-builtin inplace operations
        # you cant use this before output computation since shape calls
        # to another torch_function (infinite recursion)
        cur_node.set_input_shape(
            reduce_data_info([args, kwargs], collect_shape, [])
        )
        cur_node.set_output_shape(reduce_data_info(out, collect_shape, []))

        return out


L = TypeVar('L', bound=Iterable[Any])


def reduce_data_info(
    recorded_data: Any, action_fn: Callable[..., Any], collected: L, **kwargs: Any
) -> L:
    '''Apply action_fn to RecorderTensor inside recorded_data to collect info of
    input data into collected (Iterable) e.g. shape of RecorderTensor'''
    if isinstance(recorded_data, RecorderTensor):
        action_fn(recorded_data, collected, **kwargs)
    elif isinstance(recorded_data, Mapping):
        for r_d in recorded_data.values():
            reduce_data_info(r_d, action_fn, collected, **kwargs)
    elif (
        isinstance(recorded_data, Iterable) and
        not isinstance(recorded_data, (str, torch.Tensor))
    ):
        for r_d in recorded_data:
            reduce_data_info(r_d, action_fn, collected, **kwargs)
    return collected


def traverse_data_inplace(
    recorded_data: Any, action_fn: Callable[..., Any], **kwargs: Any
) -> None:
    '''Apply action_fn RecorderTensor objects inside recorded_data to change data
    Usuall action_fn is a function that transforms RecorderTensor in memory'''
    if isinstance(recorded_data, RecorderTensor):
        action_fn(recorded_data, **kwargs)
    elif isinstance(recorded_data, Mapping):
        for r_d in recorded_data.values():
            traverse_data_inplace(r_d, action_fn, **kwargs)
    elif (
        isinstance(recorded_data, Iterable) and
        not isinstance(recorded_data, (str, torch.Tensor))
    ):
        for r_d in recorded_data:
            traverse_data_inplace(r_d, action_fn, **kwargs)


def attach_node(
    kwargs: dict[str, Any],
    tensor_to_node: dict[RecorderTensor, TensorNode] | None = None
) -> Callable[..., Any]:
    '''Creates the function to attach TensorNodes, needed for nested calls'''
    def _func(recorded_tensor: RecorderTensor) -> None:
        '''Attaches TensorNode to ModuleNode or FunctionNode
        '''

        if kwargs['is_aux'] and tensor_to_node:
            kwargs['main_node'] = tensor_to_node[recorded_tensor]
        new_kwargs = {
            key_word: value
            for key_word, value in kwargs.items() if key_word != 'tensor_to_node'
        }
        tensor_node = TensorNode(
            tensor=recorded_tensor,
            **new_kwargs
        )
        if isinstance(kwargs["parents"], ModuleNode):
            assert getattr(recorded_tensor, 'tensor_nodes', None) is not None, (
                f'RecorderTensor to be attached to the Node'
                f'{kwargs["parents"]} must have tensor node'
            )
        assert isinstance(kwargs["parents"], (FunctionNode, ModuleNode)), (
            f'Node {kwargs["parents"]} to which to attach must be either'
            f'FunctionNode or ModuleNode'
        )

        if getattr(recorded_tensor, 'tensor_nodes', None) is None:
            recorded_tensor.tensor_nodes = [tensor_node]
        else:
            # ModuleNode: Attaches auxiliary node to tensors
            # Auxiliary nodes should be appended to keep track the node
            # history of tensor
            # FunctionNode: These should overwrite the last tensor node
            # There are 2 different cases:
            # Non-inplace ops -> New tensor and this node is the first
            # Inplace ops -> Result tensor overwrites the intput tensor
            # in memory, so it should overwrite in node history as well
            # for both cases, overwritting the last tensor node is correct
            if isinstance(kwargs["parents"], ModuleNode):
                recorded_tensor.tensor_nodes.append(tensor_node)
            elif isinstance(kwargs["parents"], FunctionNode):
                recorded_tensor.tensor_nodes[-1] = tensor_node
        kwargs["parents"].add_child(tensor_node)
        kwargs['context'].append(tensor_node)
    return _func


def pop_after_forward(
    r_in: RecorderTensor,
    recorded_output: OrderedSet[RecorderTensor],
) -> None:
    '''Removes/pops nodes from RecorderTensors to maintain correct nodes
    Two types of process exist for types of modules:
    Non-inplace ops => pop auxiliary nodes
    In-place ops => pop input nodes since inplace ops overwrites input in memory.
    '''

    in_place_func_message = (
        'Tensor before and after inplace operation must have the same memory address'
    )
    output_id: OrderedSet[int] = OrderedSet(id(x) for x in recorded_output)

    if id(r_in) not in output_id:
        _ = reduce_data_info(
            r_in, collect_tensor_node, NodeContainer(), is_pop=True
        )

    # input of inplace operation
    else:
        assert id(r_in) == r_in.tensor_nodes[-1].tensor_id, (
            in_place_func_message
        )
        assert id(r_in) == r_in.tensor_nodes[-2].tensor_id, (
            in_place_func_message
        )
        # pop tensor node before inplace operation
        r_in.tensor_nodes.pop(-2)


def collect_tensor_node(
    recorded_data: RecorderTensor,
    collected: NodeContainer[TensorNode],
    is_pop: bool = False,
) -> None:
    if getattr(recorded_data, 'tensor_nodes', None):
        if is_pop:
            collected.add(recorded_data.tensor_nodes.pop())
        else:
            collected.add(recorded_data.tensor_nodes[-1])


def collect_tensor_node_id_dict(
    recorded_data: RecorderTensor,
    collected: dict[RecorderTensor, TensorNode],
) -> None:
    if getattr(recorded_data, 'tensor_nodes', None):
        collected[recorded_data] = recorded_data.tensor_nodes[-1].main_node


def collect_tensor(
    recorded_data: RecorderTensor, collected: OrderedSet[RecorderTensor]
) -> None:
    collected.add(recorded_data)


def collect_shape(
    recorded_data: RecorderTensor, collected: list[tuple[int, ...]]
) -> None:
    collected.append(tuple(recorded_data.shape))


def process_output_node(
    cur_node: ModuleNode
) -> Callable[..., Any]:
    '''Returns function to update output node after forward
    pass of nn.Modules'''
    def _func(recorded_data: RecorderTensor) -> None:
        output_node = recorded_data.tensor_nodes[-1]
        cur_depth = cur_node.depth
        # if output node is reused inside module or empty module is used
        # introduce node for empty pass function
        if not output_node.is_leaf() or output_node.is_aux:
            insert_empty_pass_node(recorded_data, output_node)

        recorded_data.tensor_nodes[-1].depth = cur_depth
        name = 'output-tensor' if cur_depth == 0 else 'hidden-tensor'
        recorded_data.tensor_nodes[-1].name = name
        recorded_data.tensor_nodes[-1].parent_hierarchy[cur_depth] = cur_node
    return _func


def insert_empty_pass_node(
    recorded_tensor: RecorderTensor, out_node: TensorNode
) -> None:
    '''First, inserts empty-pass node as a child of tensor nodes. Then, inserts
    TensorNode as a child of this empty-pass node'''
    out_pass = FunctionNode(
        lambda x: x, out_node.depth, out_node,
        name='empty-pass'
    )
    out_node.add_child(out_pass)
    out_node.context.append(out_pass)

    passed_out_node = TensorNode(
        recorded_tensor, out_node.depth, out_pass,
        context=out_node.context, is_aux=False,
        parent_hierarchy={
            recorded_tensor.tensor_nodes[-1].depth: out_pass
        }
    )

    out_node.context.append(passed_out_node)
    out_pass.add_child(passed_out_node)

    # Update the current node of RecorderTensor
    # Here append instead of overwrite the last node because
    # this is a dummy FunctionNode that has no actual place in
    # computation graph
    recorded_tensor.tensor_nodes.append(passed_out_node)
