from __future__ import annotations

import re
import torch

from torchview.computation_node import ModuleNode, FunctionNode
from torchview.computation_graph import ComputationGraph
from torchview import draw_graph
from typing import List, Tuple, Any

from tests.fixtures.models import (
    InputNotUsed,
    MLP
)

def collect_modules_and_functions(
    model_graph: ComputationGraph
) -> Tuple[List[Tuple[str, str, str, str]], List[Tuple[str, str, str, str]]]:
    # Traverse the graph to collect the operators
    kwargs = {
        'cur_node': model_graph.node_hierarchy,
        'subgraph': None,
    }
    modules = set()
    functions = set()
    def collect_ops(**kwargs: Any) -> None:
        cur_node = kwargs['cur_node']
        if isinstance(cur_node, ModuleNode):
            signature = ", ".join([cur_node.name, str(cur_node.input_shape), str(cur_node.output_shape), str(cur_node.attributes or "n/a")])
            modules.add(signature)
        elif isinstance(cur_node, FunctionNode):
            signature = ", ".join([cur_node.name, str(cur_node.input_shape), str(cur_node.output_shape), str(cur_node.attributes or "n/a")])
            functions.add(signature)
    model_graph.traverse_graph(collect_ops, **kwargs)

    return sorted(modules), sorted(functions) # type: ignore[arg-type]

def compare_without_dtype(val1: list[Any], val2: list[Any]) -> bool:
    val1_any = {re.sub(r'dtype=\S+', 'dtype=<any>', s) for s in val1}
    val2_any = {re.sub(r'dtype=\S+', 'dtype=<any>', s) for s in val2}
    return val1_any == val2_any

def test_attributes_InputNotUsed_input_size() -> None:
    model_graph = draw_graph(
        InputNotUsed(), input_size = [(1, 128), (1, 2), (1, 2), (1, 64)],
        graph_name = 'InputNotUsed',
        expand_nested = True,
        collect_attributes = True
    )

    modules, functions  = collect_modules_and_functions(model_graph)

    # Test values
    modules_verify: list[str] = [
        'Identity, [(1, 2)], [(1, 2)], Identity(training=False)',
        'InputNotUsed, [(1, 128), (1, 2), (1, 2), (1, 64)], [(1, 2), (1, 2)], InputNotUsed(training=False)',
        'Linear, [(1, 128)], [(1, 64)], Linear(training=False, in_features=128, out_features=64)',
        'Linear, [(1, 16)], [(1, 8)], Linear(training=False, in_features=16, out_features=8)',
        'Linear, [(1, 32)], [(1, 16)], Linear(training=False, in_features=32, out_features=16)',
        'Linear, [(1, 4)], [(1, 2)], Linear(training=False, in_features=4, out_features=2)',
        'Linear, [(1, 64)], [(1, 32)], Linear(training=False, in_features=64, out_features=32)',
        'Linear, [(1, 8)], [(1, 4)], Linear(training=False, in_features=8, out_features=4)',
        'ReLU, [(1, 16)], [(1, 16)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 2)], [(1, 2)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 32)], [(1, 32)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 4)], [(1, 4)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 64)], [(1, 64)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 8)], [(1, 8)], ReLU(training=False, inplace=True)',
        'Sequential, [(1, 128)], [(1, 2)], Sequential(training=False)'
    ]

    functions_verify: list[str] = [
        'add, [(1, 2), (1, 2)], [(1, 2)], [[Tensor(shape=(1, 2), dtype=torch.float32), Tensor(shape=(1, 2), dtype=torch.float32)], {}]'
    ]

    # Ignore dtype in the comparison, as it can vary
    assert compare_without_dtype(modules, modules_verify)
    assert compare_without_dtype(functions, functions_verify)

def test_no_attributes_InputNotUsed_input_size() -> None:
    model_graph = draw_graph(
        InputNotUsed(), input_size = [(1, 128), (1, 2), (1, 2), (1, 64)],
        graph_name = 'InputNotUsed',
        expand_nested = True,
    )

    modules, functions  = collect_modules_and_functions(model_graph)

    # Test values
    modules_verify: list[str] = [
        'Identity, [(1, 2)], [(1, 2)], n/a',
        'InputNotUsed, [(1, 128), (1, 2), (1, 2), (1, 64)], [(1, 2), (1, 2)], n/a',
        'Linear, [(1, 128)], [(1, 64)], n/a',
        'Linear, [(1, 16)], [(1, 8)], n/a',
        'Linear, [(1, 32)], [(1, 16)], n/a',
        'Linear, [(1, 4)], [(1, 2)], n/a',
        'Linear, [(1, 64)], [(1, 32)], n/a',
        'Linear, [(1, 8)], [(1, 4)], n/a',
        'ReLU, [(1, 16)], [(1, 16)], n/a',
        'ReLU, [(1, 2)], [(1, 2)], n/a',
        'ReLU, [(1, 32)], [(1, 32)], n/a',
        'ReLU, [(1, 4)], [(1, 4)], n/a',
        'ReLU, [(1, 64)], [(1, 64)], n/a',
        'ReLU, [(1, 8)], [(1, 8)], n/a',
        'Sequential, [(1, 128)], [(1, 2)], n/a'
    ]

    functions_verify: list[str] = [
        'add, [(1, 2), (1, 2)], [(1, 2)], n/a'
    ]

    # Ignore dtype in the comparison, as it can vary
    assert compare_without_dtype(modules, modules_verify)
    assert compare_without_dtype(functions, functions_verify)


def test_attributes_MLP_input_size() -> None:
    model_graph = draw_graph(
        MLP(), input_size = (1, 128),
        graph_name = 'MLP',
        collect_attributes = True
    )

    modules, functions  = collect_modules_and_functions(model_graph)

    # Test values
    modules_verify: list[str] = [
        'Linear, [(1, 128)], [(1, 64)], Linear(training=False, in_features=128, out_features=64)',
        'Linear, [(1, 16)], [(1, 8)], Linear(training=False, in_features=16, out_features=8)',
        'Linear, [(1, 32)], [(1, 16)], Linear(training=False, in_features=32, out_features=16)',
        'Linear, [(1, 4)], [(1, 2)], Linear(training=False, in_features=4, out_features=2)',
        'Linear, [(1, 64)], [(1, 32)], Linear(training=False, in_features=64, out_features=32)',
        'Linear, [(1, 8)], [(1, 4)], Linear(training=False, in_features=8, out_features=4)',
        'MLP, [(1, 128)], [(1, 2)], MLP(training=False)',
        'ReLU, [(1, 16)], [(1, 16)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 32)], [(1, 32)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 4)], [(1, 4)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 64)], [(1, 64)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 8)], [(1, 8)], ReLU(training=False, inplace=True)',
        'Sequential, [(1, 128)], [(1, 2)], Sequential(training=False)'
    ]

    functions_verify: list[str] = []

    # Ignore dtype in the comparison, as it can vary
    assert compare_without_dtype(modules, modules_verify)
    assert compare_without_dtype(functions, functions_verify)

def test_attributes_MLP_input_tensor() -> None:
    input_tensor = torch.rand(1, 128)
    model_graph = draw_graph(
        MLP(), input_tensor,
        graph_name = 'MLP',
        collect_attributes = True
    )

    modules, functions  = collect_modules_and_functions(model_graph)

    # Test values
    modules_verify: list[str] = [
        'Linear, [(1, 128)], [(1, 64)], Linear(training=False, in_features=128, out_features=64)',
        'Linear, [(1, 16)], [(1, 8)], Linear(training=False, in_features=16, out_features=8)',
        'Linear, [(1, 32)], [(1, 16)], Linear(training=False, in_features=32, out_features=16)',
        'Linear, [(1, 4)], [(1, 2)], Linear(training=False, in_features=4, out_features=2)',
        'Linear, [(1, 64)], [(1, 32)], Linear(training=False, in_features=64, out_features=32)',
        'Linear, [(1, 8)], [(1, 4)], Linear(training=False, in_features=8, out_features=4)',
        'MLP, [(1, 128)], [(1, 2)], MLP(training=False)',
        'ReLU, [(1, 16)], [(1, 16)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 32)], [(1, 32)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 4)], [(1, 4)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 64)], [(1, 64)], ReLU(training=False, inplace=True)',
        'ReLU, [(1, 8)], [(1, 8)], ReLU(training=False, inplace=True)',
        'Sequential, [(1, 128)], [(1, 2)], Sequential(training=False)'
    ]

    functions_verify: list[str] = []

    # Ignore dtype in the comparison, as it can vary
    assert compare_without_dtype(modules, modules_verify)
    assert compare_without_dtype(functions, functions_verify)

def test_no_attributes_MLP_input_tensor() -> None:
    input_tensor = torch.rand(1, 128)
    model_graph = draw_graph(
        MLP(), input_tensor,
        graph_name = 'MLP',
    )

    modules, functions  = collect_modules_and_functions(model_graph)

    # Test values
    modules_verify: list[str] = [
        'Linear, [(1, 128)], [(1, 64)], n/a',
        'Linear, [(1, 16)], [(1, 8)], n/a',
        'Linear, [(1, 32)], [(1, 16)], n/a',
        'Linear, [(1, 4)], [(1, 2)], n/a',
        'Linear, [(1, 64)], [(1, 32)], n/a',
        'Linear, [(1, 8)], [(1, 4)], n/a',
        'MLP, [(1, 128)], [(1, 2)], n/a',
        'ReLU, [(1, 16)], [(1, 16)], n/a',
        'ReLU, [(1, 32)], [(1, 32)], n/a',
        'ReLU, [(1, 4)], [(1, 4)], n/a',
        'ReLU, [(1, 64)], [(1, 64)], n/a',
        'ReLU, [(1, 8)], [(1, 8)], n/a',
        'Sequential, [(1, 128)], [(1, 2)], n/a'
    ]

    functions_verify: list[str] = []

    # Ignore dtype in the comparison, as it can vary
    assert compare_without_dtype(modules, modules_verify)
    assert compare_without_dtype(functions, functions_verify)
