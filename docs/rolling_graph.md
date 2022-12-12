# How to identify nodes and roll recursive graphs ?

## TODO: Add Images, clear explanation

In this document, we describe the main mechanism used to identify the computations nodes (as being the same when rolling recursive nodes) and used to
display accordingly.

Firstly, note that the main information used to identify nodes is the `node_id` property of ComputationNodes. For instance, if the node_id of two nodes
(probably recursivly used) are the same, they are shown with the same node on the visual graph.

Regarding the issue of rolling mechanism, there are simply two modes:

* `Roll=False`
* `Roll=True`

If `Roll=False`, each computation step is displayed on the graph uniquely despite whether they are recursively used or not. It is as simple as that. This
is done by setting node_id of each node to python id of ComputationNodes (e.g. node_id=id(tensor_node_1)).

If `Roll=True`, then we identify "recursively" used modules as being the same on the visual graph. The mechanism to identify nodes depends on type of nodes.

1. **TensorNode:** TensorNodes are not rolled at all, and they all are uniquely identified. This choice is made since each tensor is used uniquely unlike modules
in recursive networks

2. **FunctionNode:** Functions are identified by the id of torch function associated with output of this FunctionNode. If these are the same, then two FunctionNodes are identified.
The reason behinc choice is as follows. First of all, we cannot only use id of torch function since, for instance, two torch functions used in different places but are the same torch function would be identified (e.g. two skip connection used in different places would lead torch.add in different places). But, visually this is not appealing at all. Imagine two skip connection (far apart) sharing the same node, very ugly.
But, there is the choice of input id. Well, this is not really optimal either. Image, you have one tensor and and that this tensor is applied to two different torch.relu calls. You would want to see 2 distinct branch of function nodes coming out of this input node. But, if you identify with input nodes, then there will be only one branch. This is because identification node (input node) is the same for each torch.relu call.
Finally. we are left with the choice of output node id. This perfectly makes sense since each output tensor must originate from a unique function call. Meaning, once you tell me the tensor node, I can uniquely tell you function node that it returned. For instance, this leads to 2 distinct branches in the previous example of `torch.relu`.
There is another thing to consider when we hide the inner tensors. It is that output to identify to nodes by wont be `TensorNodes`. Above, we argued that it all works well in the case identification by output tensornode. The mechanism still works well for output Module or FunctionNodes. The reason for this is similar to the previous argument (input node id does not work and output node id gives your unique way to identify FunctionNode).

3. **ModuleNode:** For ModuleNode, there two subcases: Stateless and Stateful Modules.
Stateless Module refers to any module that has no `torch.nn.parameter.Parameter`. Usually, the way these modules are used is that they are created once and used in different places and multiple times. So, they pretty much used as a torch function. Therefore, they are identified the same way as FunctionNode. To convince you even more, imagine that they are identified by output id but rather the id of python object itself (like stateful module, see below), then the same e.g. all the ReLUs would be connected to the one shared node, (quite ugly).
Secondly, we look at Stateful Module. These are idenfitied by the id of the python object ModuleNode. This is very reasonable if you want to roll the graph, you want to the object (i.e. Module with the same parameters) to appear the same visually.
