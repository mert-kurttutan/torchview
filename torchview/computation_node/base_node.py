from __future__ import annotations

from ..utils import OrderedSet

# arr_type is the fundemental data structure to store parent and
# outputs nodes. For normal usage, the set is used. This seemed most
# appropriate data structure bec. parent and outputs nodes have to be
# unique. If they are not unique, this leads to multiple edges between nodes
# which I dont see as reasonable in any representation of torch models
# Other proposals are welcome. One drawback is the order of iteration of set is not
# deterministic. This is important to get reproducible result. To do this,
# used OrderedSet see utils.py. For instance, when producing test result change
# arr_type to OrderedSet to get deterministic results.
NodeContainer = OrderedSet

# TODO: find a better a way to set arr_type intead of manually
# changing during development.
# TODO: Benchmark between set and OrderedSet implementation
# One benchmark on bencmark-v1.py file:
# set: 3.71 ± 0.05
# OrderedSet: 3.89 ± 0.10 => 6 % more time


class Node:
    '''Base Class Node to keep track of Computation Graph of torch models'''
    def __init__(
        self,
        depth: int,
        parents: NodeContainer[Node] | Node | None = None,
        outputs: NodeContainer[Node] | Node | None = None,
        name: str = 'node',
    ) -> None:
        if outputs is None:
            outputs = NodeContainer()
        if parents is None:
            parents = NodeContainer()

        self.outputs = (
            NodeContainer([outputs]) if isinstance(outputs, Node)
            else outputs
        )

        self.parents = (
            NodeContainer([parents]) if isinstance(parents, Node)
            else parents
        )

        self.name = name
        self.depth = depth
        self.node_id = 'null'

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.name} at {hex(id(self))}"

    def add_outputs(self, node: Node) -> None:
        self.outputs.add(node)

    def add_parents(self, node: Node) -> None:
        self.parents.add(node)

    def remove_outputs(self, node: Node) -> None:
        self.outputs.remove(node)

    def remove_parents(self, node: Node) -> None:
        self.parents.remove(node)

    def set_outputs(self, node_arr: NodeContainer[Node]) -> None:
        self.outputs = node_arr

    def set_parents(self, node_arr: NodeContainer[Node]) -> None:
        self.parents = node_arr

    def is_main_parent(self) -> bool:
        return not self.parents

    def is_main_output(self) -> bool:
        return not self.outputs

    def set_node_id(self) -> None:
        raise NotImplementedError(
            'To be implemented by subclasses of Node Class !!!'
        )
