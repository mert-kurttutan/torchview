from __future__ import annotations

from ..utils import OrderedSet

# arr_type is the fundemental data structure to store parent and
# children nodes. For normal usage, the set is used. This seemed most
# appropriate data structure bec. parent and children nodes have to be
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
        children: NodeContainer[Node] | Node | None = None,
        name: str = 'node',
    ) -> None:
        if children is None:
            children = NodeContainer()
        if parents is None:
            parents = NodeContainer()

        self.children = (
            NodeContainer([children]) if isinstance(children, Node)
            else children
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

    def add_children(self, node: Node) -> None:
        self.children.add(node)

    def add_parents(self, node: Node) -> None:
        self.parents.add(node)

    def remove_children(self, node: Node) -> None:
        self.children.remove(node)

    def remove_parents(self, node: Node) -> None:
        self.parents.remove(node)

    def set_children(self, node_arr: NodeContainer[Node]) -> None:
        self.children = node_arr

    def set_parents(self, node_arr: NodeContainer[Node]) -> None:
        self.parents = node_arr

    def remove(self) -> None:
        '''Removes the node from its graph by connecting its
        parents to its children. Special cases e.g. no parent node
        will be considered
        '''
        if len(self.children) > 1 and len(self.parents) > 1:
            parents_list = list(self.parents)
            for parent_node in parents_list:
                if parent_node.depth == -10:
                    parent_node.remove_children(self)
                    self.remove_parents(parent_node)

            assert len(self.parents) == 1, (
                f'{self}'
            )
        # if it has no parent node
        if not self.parents:
            for children_node in self.children:
                children_node.remove_parents(self)

        else:
            # if it has no children node
            if not self.children:
                for parent_node in self.parents:
                    parent_node.remove_children(self)
            else:
                for parent_node in self.parents:
                    parent_node.remove_children(self)
                    for children_node in self.children:
                        parent_node.add_children(children_node)
                        children_node.add_parents(parent_node)

                for j in self.children:
                    j.remove_parents(self)

        # nullify the content
        self.set_children(NodeContainer())
        self.set_parents(NodeContainer())
        self.name = ''
        self.node_id = 'null'

    def set_node_id(self) -> None:
        raise NotImplementedError(
            'To be implemented by subclasses of Node Class !!!'
        )
