import re

from torchview.computation_node import ModuleNode, FunctionNode
from torchview import draw_graph
from tests.fixtures.models import (
    InputNotUsed,
    MLP
)

def collect_modules_and_functions(
    model_graph
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    # Traverse the graph to collect the operators
    kwargs = {
        'cur_node': model_graph.node_hierarchy,
        'subgraph': None,
    }
    modules = set()
    functions = set()
    def collect_ops(**kwargs) -> None:
        cur_node = kwargs['cur_node']
        if isinstance(cur_node, ModuleNode):
            signature = ", ".join([cur_node.name, str(cur_node.input_shape), str(cur_node.output_shape), cur_node.attributes])
            modules.add(signature)
        elif isinstance(cur_node, FunctionNode):
            signature = ", ".join([cur_node.name, str(cur_node.input_shape), str(cur_node.output_shape), cur_node.attributes])
            functions.add(signature)
    model_graph.traverse_graph(collect_ops, **kwargs)

    return sorted(modules), sorted(functions)

def compare_without_dtype(val1: set, val2: set) -> bool:
    val1 = {re.sub(r'dtype=\S+', 'dtype=<any>', s) for s in val1}
    val2 = {re.sub(r'dtype=\S+', 'dtype=<any>', s) for s in val2}
    return val1 == val2

def test_attributes_InputNotUsed() -> None:
    model_graph = draw_graph(
        InputNotUsed(), input_size=[(1, 128), (1, 2), (1, 2), (1, 64)],
        graph_name='InputNotUsed',
        expand_nested=True,
    )

    modules, functions  = collect_modules_and_functions(model_graph)

    # Test values
    modules_verify = [
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

    functions_verify = [
        'add, [(1, 2), (1, 2)], [(1, 2)], [Tensor(shape=(1, 2), dtype=torch.bfloat16), Tensor(shape=(1, 2), dtype=torch.bfloat8)]'
    ]

    # Ignore dtype in the comparison, as it can vary
    assert compare_without_dtype(modules, modules_verify)
    assert compare_without_dtype(functions, functions_verify)


def test_attributes_MLP() -> None:
    model_graph = draw_graph(
        MLP(), input_size=(1, 128),
        graph_name='MLP',
    )

    modules, functions  = collect_modules_and_functions(model_graph)

    # Test values
    modules_verify = [
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

    functions_verify = []

    # Ignore dtype in the comparison, as it can vary
    assert compare_without_dtype(modules, modules_verify)
    assert compare_without_dtype(functions, functions_verify)
