from __future__ import annotations

from ..utils import OrderedSet

# arr_type is the fundemental data structure to store input and
# outputs nodes. For normal usage, the set is used. This seemed most
# appropriate data structure bec. input and outputs nodes have to be
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
        inputs: NodeContainer[Node] | Node | None = None,
        outputs: NodeContainer[Node] | Node | None = None,
        name: str = 'node',
        children: NodeContainer[Node] | Node | None = None,
    ) -> None:
        if outputs is None:
            outputs = NodeContainer()
        if inputs is None:
            inputs = NodeContainer()

        self.outputs = (
            NodeContainer([outputs]) if isinstance(outputs, Node)
            else outputs
        )

        self.inputs = (
            NodeContainer([inputs]) if isinstance(inputs, Node)
            else inputs
        )

        self.children = (
            NodeContainer([children]) if isinstance(children, Node)
            else children
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

    def add_inputs(self, node: Node) -> None:
        self.inputs.add(node)

    def remove_outputs(self, node: Node) -> None:
        self.outputs.remove(node)

    def remove_inputs(self, node: Node) -> None:
        self.inputs.remove(node)

    def set_outputs(self, node_arr: NodeContainer[Node]) -> None:
        self.outputs = node_arr

    def set_inputs(self, node_arr: NodeContainer[Node]) -> None:
        self.inputs = node_arr

    def remove(self) -> None:
        '''Removes the node from its graph by connecting its
        inputs to its outputs. Special cases e.g. no input node
        will be considered
        '''
        if len(self.outputs) > 1 and len(self.inputs) > 1:
            inputs_list = list(self.inputs)
            for input_node in inputs_list:
                if input_node.depth == -10:
                    input_node.remove_outputs(self)
                    self.remove_inputs(input_node)

        # if it has no input node
        if not self.inputs:
            for outputs_node in self.outputs:
                outputs_node.remove_inputs(self)

        else:
            # if it has no outputs node
            if not self.outputs:
                for input_node in self.inputs:
                    input_node.remove_outputs(self)
            else:
                for input_node in self.inputs:
                    input_node.remove_outputs(self)
                    for outputs_node in self.outputs:
                        input_node.add_outputs(outputs_node)
                        outputs_node.add_inputs(input_node)

                for j in self.outputs:
                    j.remove_inputs(self)

        # nullify the content
        self.set_outputs(NodeContainer())
        self.set_inputs(NodeContainer())
        self.name = ''
        self.node_id = 'null'

    def set_node_id(self) -> None:
        raise NotImplementedError(
            'To be implemented by subclasses of Node Class !!!'
        )
