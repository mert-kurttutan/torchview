from __future__ import annotations

from typing import Any, Iterable, Mapping, TypeVar
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .computation_node import ModuleNode, FunctionNode, TensorNode, NodeContainer

# Needed for module wrapper and resetting
_orig_module_forward = torch.nn.Module.__call__


# Functions below are torch.tensor creation operations
# Inside Recorder context, we wrapped these functions
# so that these return RecorderTensor can be recorded during
# forward propagation
orig_name_list = [
    'randn', 'ones', 'range', 'linspace', 'logspace',
    'zeros', 'randint', 'rand', 'rand_like', 'normal',
    'bernoulli', 'multinomial', 'poisson', 'randint',
]
_orig_op_list = [getattr(torch, name) for name in orig_name_list]


class Recorder:
    '''Context Manager that sets modules forward and torch creation ops
    to record them in computation graph'''
    def __init__(
        self, orig_mod_forward: Callable[..., Any], new_mod_forward: Callable[..., Any]
    ) -> None:
        self.orig_module_forward = orig_mod_forward
        self.new_module_forward = new_mod_forward

    def __enter__(self) -> None:
        setattr(
            torch.nn.Module, "__call__", self.new_module_forward
        )

        for name, op in zip(orig_name_list, _orig_op_list):
            setattr(
                torch, name, creation_ops_wrapper(op)
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


def creation_ops_wrapper(_orig_op: Callable[..., Any]) -> Callable[..., Any]:
    def _func(*args: Any, **kwargs: Any) -> RecorderTensor:

        input_tensor = _orig_op(*args, **kwargs)

        input_recorder_tensor: RecorderTensor = input_tensor.as_subclass(RecorderTensor)
        input_recorder_tensor.tensor_nodes = []
        input_node = TensorNode(
            tensor=input_recorder_tensor,
            depth=-10,
            name='temporary-tensor',
        )

        input_recorder_tensor.tensor_nodes.append(input_node)
        return input_recorder_tensor
    return _func


def module_forward_wrapper(
    hide_module_functions: bool,
) -> Callable[..., Any]:
    '''Wrapper for forward functions of modules'''
    def _module_forward_wrapper(mod: nn.Module, *args: Any, **kwargs: Any) -> Any:
        '''Forward prop of module for RecorderTensor subclass
        Construct Module Node => forward-prop => change output nodes to retain
        module hierarchy correctly
        '''
        # Create module node and connect to its parents tensor node
        input_nodes: NodeContainer[TensorNode] = (
            reduce_data_info([args, kwargs], collect_tensor_node, NodeContainer())
        )
        # if none of args originated from input
        # hence only torch.Tensor
        if not input_nodes:
            return _orig_module_forward(mod, *args, **kwargs)

        # Create module_node and connect to its parents tensor node
        cur_depth = next(iter(input_nodes)).depth
        cur_node = ModuleNode(
            mod, cur_depth, input_nodes,  # type: ignore[arg-type]
            name=type(mod).__name__
        )
        for node in input_nodes:
            node.add_children(cur_node)
            node.name = "hidden-tensor"
        cur_node.set_input_shape(
            reduce_data_info([args, kwargs], collect_shape, [])
        )
        traverse_data_inplace(
            [args, kwargs], attach_node(cur_node, cur_node.depth+1)
        )

        # TODO: check if output contains RecorderTensor
        # this seems not to be necessary so far
        out = _orig_module_forward(mod, *args, **kwargs)

        traverse_data_inplace(
            out, change_depth_name(-1, 'output-tensor')
        )

        # pop appropriate nodes, see implementation below
        input_recorder: NodeContainer[RecorderTensor] = (
            reduce_data_info([args, kwargs], collect_tensor, NodeContainer())
        )
        output_recorder: NodeContainer[RecorderTensor] = (
            reduce_data_info(out, collect_tensor, NodeContainer())
        )
        pop_after_forward(input_recorder, output_recorder)

        # remove auxiliary tensor nodes
        output_nodes: NodeContainer[TensorNode] = (
            reduce_data_info(out, collect_tensor_node, NodeContainer())
        )
        auxiliary_nodes = list(cur_node.children)
        for aux_node in auxiliary_nodes:
            assert isinstance(aux_node, TensorNode), (
                f'Auxiliary Node of the module node'
                f'{cur_node} must be a Tensor Node!'
            )
            # some modules are auxiliary modules, just passes
            # without any computation. e.g. nn.Identity(). For these modules,
            # output node is the same as auxuliary nodes, so dont delete the nodes
            # if this is the case
            if aux_node not in output_nodes:
                aux_node.remove()

        # dont touch inner tensors here, it might disrupt
        # rest of the algo that depends previous tensors
        if hide_module_functions and not any(mod.children()):
            # keep removing until all output tensor nodes
            # are children of current module node
            while not (
                    reduce_data_info(out, collect_tensor_node, NodeContainer())
                    .issubset(cur_node.children)  # type: ignore[arg-type]
            ):
                remove_func_module(out, cur_node)

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
        cls: Any, func: Callable[..., Any],
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
        if not reduce_data_info(out, collect_tensor, NodeContainer()):
            return out

        # Create function_node and connect to its parents tensor node
        cur_depth = next(iter(args_nodes)).depth
        cur_node = FunctionNode(
            func, cur_depth, args_nodes, name=func.__name__  # type: ignore[arg-type]
        )
        for i in args_nodes:
            i.add_children(cur_node)
            i.name = 'hidden-tensor'

        traverse_data_inplace(out, attach_node(cur_node))

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
            traverse_data_inplace(r_d, action_fn)
    elif (
        isinstance(recorded_data, Iterable) and
        not isinstance(recorded_data, (str, torch.Tensor))
    ):
        for r_d in recorded_data:
            traverse_data_inplace(r_d, action_fn)


def attach_node(
    parent_node: FunctionNode | ModuleNode, depth: int | None = None
) -> Callable[..., Any]:
    '''Creates the function to attach TensorNodes, needed for nested calls'''
    def _func(recorded_tensor: RecorderTensor) -> None:
        '''Attaches TensorNode to ModuleNode or FunctionNode
        '''
        _depth = parent_node.depth if depth is None else depth

        tensor_node = TensorNode(
            tensor=recorded_tensor,
            depth=_depth,
            parents=parent_node,
        )
        if isinstance(parent_node, ModuleNode):
            assert getattr(recorded_tensor, 'tensor_nodes', None) is not None, (
                f'RecorderTensor to be attached to the Node'
                f'{parent_node} must have tensor node'
            )
        assert isinstance(parent_node, (FunctionNode, ModuleNode)), (
            f'Node {parent_node} to which to attach must be either'
            f'FunctionNode or ModuleNode'
        )

        if getattr(recorded_tensor, 'tensor_nodes', None) is None:
            recorded_tensor.tensor_nodes = [tensor_node]
        else:
            if isinstance(parent_node, ModuleNode):
                recorded_tensor.tensor_nodes.append(tensor_node)
            elif isinstance(parent_node, FunctionNode):
                recorded_tensor.tensor_nodes[-1] = tensor_node
        parent_node.add_children(tensor_node)
    return _func


def pop_after_forward(
    recorded_input: NodeContainer[RecorderTensor],
    recorded_output: NodeContainer[RecorderTensor]
) -> None:
    '''Removes/pops nodes from RecorderTensors to maintain correct nodes
    Two types of process exist for types of modules:
    Non-inplace ops => pop auxiliary nodes
    In-place ops => pop input nodes since inplace ops overwrites input in memory.
    '''

    in_place_func_message = (
        'Tensor before and after inplace operation must have the same memory address'
    )
    output_id: NodeContainer[int] = NodeContainer(id(x) for x in recorded_output)
    for r_in in recorded_input:
        if not id(r_in) in output_id:
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


def remove_func_module(out: Any, mod_node: ModuleNode) -> None:
    '''Removes all parent nodes of RecorderTensor Nodes
    from the graph
    '''
    output_tensor_nodes: NodeContainer[TensorNode] = (
        reduce_data_info(out, collect_tensor_node, NodeContainer())
    )
    my_col: NodeContainer[TensorNode] = NodeContainer()

    # collect all parent nodes
    for out_node in output_tensor_nodes:
        for parent in out_node.parents:
            my_col.add(parent)  # type: ignore[arg-type]

    for par in my_col:
        assert not isinstance(par, ModuleNode), (
            f'Module Node {mod_node} has no children module,'
            f'hence must not have node of Module Node type {par}'
        )
        par.remove()


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


def collect_tensor(
    recorded_data: RecorderTensor, collected: NodeContainer[RecorderTensor]
) -> None:
    collected.add(recorded_data)


def collect_shape(
    recorded_data: RecorderTensor, collected: list[tuple[int, ...]]
) -> None:
    collected.append(tuple(recorded_data.shape))


def change_depth_name(depth_delta: int, name: str) -> Callable[..., Any]:
    def _func(recorded_data: RecorderTensor) -> None:
        recorded_data.tensor_nodes[-1].depth += depth_delta
        recorded_data.tensor_nodes[-1].name = name
    return _func
