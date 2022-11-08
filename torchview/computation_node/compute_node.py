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
        parents: NodeContainer[Node] | Node | None = None,
        children: NodeContainer[Node] | Node | None = None,
        name: str = 'tensor',
    ):

        super(TensorNode, self).__init__(
            depth, parents, children, name,
        )
        self.tensor_id = id(tensor)
        self.tensor_shape = tuple(tensor.shape)
        self.name = name
        self.set_node_id()

    def set_node_id(self, parent_id: int | str | None = None) -> None:
        if parent_id is None:
            self.node_id = f'{id(self)}'
        else:
            self.node_id = f'{id(self)}-{parent_id}'


class ModuleNode(Node):
    '''Subclass of node specialzed for storing torch Module info
    '''
    def __init__(
        self,
        module_unit: nn.Module,
        depth: int,
        parents: NodeContainer[Node] | Node | None = None,
        children: NodeContainer[Node] | Node | None = None,
        name: str = 'module-node',
    ) -> None:
        super(ModuleNode, self).__init__(
            depth, parents, children, name
        )
        self.compute_unit_id = id(module_unit)
        self.is_activation = is_generator_empty(module_unit.parameters())
        self.input_shape: list[Tuple[int, ...]] = []
        self.output_shape: list[Tuple[int, ...]] = []
        self.set_node_id()

    def set_input_shape(self, input_shape: list[Tuple[int, ...]]) -> None:
        self.input_shape = input_shape

    def set_output_shape(self, output_shape: list[Tuple[int, ...]]) -> None:
        self.output_shape = output_shape

    def set_node_id(self, child_id: int | str | None = None) -> None:
        '''Sets the id of ModuleNode.
        If no child is given, it sets to value unique to node.
        If child id is given, there are 2 cases:
            1. Parameterless module: id is determined by child_id and id of nn.Module
            2. Module with parameter: id is determined by only id of nn.module object
        This is crucial when rolling recursive modules by identifying them with this id
        mechanism'''
        if child_id is None:
            self.node_id = f'{id(self)}'
        else:
            if self.is_activation:
                # zero-parameter module -> module for activation function, e.g. ReLU
                self.node_id = f'{self.compute_unit_id}-{child_id}'
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
        parents: NodeContainer[Node] | Node | None = None,
        children: NodeContainer[Node] | Node | None = None,
        name: str = 'function-node',
    ) -> None:
        super(FunctionNode, self).__init__(
            depth, parents, children, name
        )
        self.compute_unit_id = id(function_unit)
        self.input_shape: list[Tuple[int, ...]] = []
        self.output_shape: list[Tuple[int, ...]] = []
        self.set_node_id()

    def set_input_shape(self, input_shape: list[Tuple[int, ...]]) -> None:
        self.input_shape = input_shape

    def set_output_shape(self, output_shape: list[Tuple[int, ...]]) -> None:
        self.output_shape = output_shape

    def set_node_id(self, child_id: int | str | None = None) -> None:
        '''Sets the id of FunctionNode.
        If no child is given, it sets to value unique to node.
        If child id is given, id is determined by only id of nn.module object
        This is crucial when rolling recursive modules by identifying them with this id
        mechanism'''
        if child_id is None:
            self.node_id = f'{id(self)}'
        else:
            self.node_id = f'{self.compute_unit_id}-{child_id}'
