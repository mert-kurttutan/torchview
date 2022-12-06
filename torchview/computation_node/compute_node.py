from __future__ import annotations

from typing import Tuple, Any
from collections.abc import Callable

import torch
from torch import nn

from .base_node import Node, NodeContainer
from ..utils import is_generator_empty


class TensorNode(Node):
    '''Subclass of node specialzed for nodes that
    stores tensor (subclass of torch.Tensor called RecorderTensor)
    '''
    def __init__(
        self,
        tensor: torch.Tensor,
        depth: int,
        inputs: NodeContainer[Node] | Node | None = None,
        outputs: NodeContainer[Node] | Node | None = None,
        name: str = 'tensor',
        context: Any | None = None,
        is_aux: bool = False,
        main_node: TensorNode | None = None,
        input_hierarchy: dict[int, ModuleNode] | None = None,
    ):

        super(TensorNode, self).__init__(
            depth, inputs, outputs, name,
        )
        self.tensor_id = id(tensor)
        self.tensor_shape = tuple(tensor.shape)
        self.name = name
        self.is_aux = is_aux
        self.main_node = self if main_node is None else main_node
        self.context = [] if context is None else context
        self.input_hierarchy = {} if input_hierarchy is None else input_hierarchy
        self.set_node_id()

    def set_node_id(self, input_id: int | str | None = None) -> None:
        if input_id is None:
            self.node_id =  f'{id(self.main_node)}' if self.is_aux and self.main_node else f'{id(self)}'
        else:
            self.node_id = f'{id(self)}-{input_id}'


class ModuleNode(Node):
    '''Subclass of node specialzed for storing torch Module info
    '''
    def __init__(
        self,
        module_unit: nn.Module,
        depth: int,
        inputs: NodeContainer[Node] | Node | None = None,
        outputs: NodeContainer[Node] | Node | None = None,
        name: str = 'module-node',
        end_nodes: NodeContainer[Node] | None = None,
    ) -> None:
        super(ModuleNode, self).__init__(
            depth, inputs, outputs, name
        )
        self.compute_unit_id = id(module_unit)
        self.is_activation = is_generator_empty(module_unit.parameters())
        self.is_container = not any(module_unit.children())
        self.input_shape: list[Tuple[int, ...]] = []
        self.output_shape: list[Tuple[int, ...]] = []
        self.end_nodes = NodeContainer() if end_nodes is None else end_nodes
        self.set_node_id()

    def set_input_shape(self, input_shape: list[Tuple[int, ...]]) -> None:
        self.input_shape = input_shape

    def set_output_shape(self, output_shape: list[Tuple[int, ...]]) -> None:
        self.output_shape = output_shape

    def set_node_id(self, output_id: int | str | None = None) -> None:
        '''Sets the id of ModuleNode.
        If no output is given, it sets to value unique to node.
        If output id is given, there are 2 cases:
            1. Parameterless module: id is determined by output_id and id of nn.Module
            2. Module with parameter: id is determined by only id of nn.module object
        This is crucial when rolling recursive modules by identifying them with this id
        mechanism'''
        if output_id is None:
            self.node_id = f'{id(self)}'
        else:
            if self.is_activation:
                # zero-parameter module -> module for activation function, e.g. ReLU
                self.node_id = f'{self.compute_unit_id}-{output_id}'
            else:
                self.node_id = f'{self.compute_unit_id}-'


class FunctionNode(Node):
    '''Subclass of node specialized for nodes
    that does computation (e.g. torch.functions)
    '''
    def __init__(
        self,
        function_unit: Callable[..., Any],
        depth: int,
        inputs: NodeContainer[Node] | Node | None = None,
        outputs: NodeContainer[Node] | Node | None = None,
        name: str = 'function-node',
    ) -> None:
        super(FunctionNode, self).__init__(
            depth, inputs, outputs, name
        )
        self.compute_unit_id = id(function_unit)
        self.is_container = True
        self.input_shape: list[Tuple[int, ...]] = []
        self.output_shape: list[Tuple[int, ...]] = []
        self.set_node_id()
        self.end_nodes = self.outputs

    def set_input_shape(self, input_shape: list[Tuple[int, ...]]) -> None:
        self.input_shape = input_shape

    def set_output_shape(self, output_shape: list[Tuple[int, ...]]) -> None:
        self.output_shape = output_shape

    def set_node_id(self, output_id: int | str | None = None) -> None:
        '''Sets the id of FunctionNode.
        If no output is given, it sets to value unique to node.
        If output id is given, id is determined by only id of nn.module object
        This is crucial when rolling recursive modules by identifying them with this id
        mechanism'''
        if output_id is None:
            self.node_id = f'{id(self)}'
        else:
            self.node_id = f'{self.compute_unit_id}-{output_id}'
