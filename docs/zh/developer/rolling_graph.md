# 如何识别节点并折叠递归图（Rolling）？

## TODO（原文保留）

TODO: 添加图片与更清晰的解释

## 概述

本文档描述 torchview 在 **rolling（递归折叠）** 场景下，如何识别计算节点“是否应当被视为同一个节点”，以及最终如何呈现在可视化图中。

首先需要说明：用于识别节点的核心信息是 `ComputationNode` 的 `node_id` 属性。举例来说，如果两个节点（通常来自递归重复使用）拥有相同的 `node_id`，它们会在最终的可视化图中合并为同一个节点。

## Rolling 的两种模式

rolling 机制主要只有两种模式：

- `roll=False`
- `roll=True`

### `roll=False`

当 `roll=False` 时，每一次计算步骤都会在图中**唯一展示**（不管它是否来自递归复用）。实现方式很简单：将每个节点的 `node_id` 设为该 `ComputationNode` Python 对象的 id（例如 `node_id = id(tensor_node_1)`）。

### `roll=True`

当 `roll=True` 时，我们会将“递归复用的模块”在可视化图里识别为同一个节点。具体如何识别，取决于节点类型：

1. **TensorNode**

TensorNode **不会被折叠**，每个 tensor 都会被唯一识别。这是因为 tensor 通常不像递归模块那样“重复使用同一个对象”，并且把 tensor 折叠会让图更难理解。

2. **FunctionNode**

FunctionNode 的识别基于：与该 FunctionNode 的 **输出**相关联的 torch 函数的 id。如果这些相同，则两个 FunctionNode 会被视为同一个。

为什么要用“输出”来识别？原因可以分几种讨论：

- **不能只用 torch 函数自身的 id**：例如两处不同位置都调用了 `torch.add`（不同跳连），如果只按函数 id 合并，会把相距很远的结构合并到一个节点上，视觉效果很差。
- **用输入来识别也不理想**：设想同一个输入 tensor 分别走向两个不同的 `torch.relu` 调用。我们希望图中看到两个分支。但如果按输入识别，会导致只出现一个分支（因为输入相同）。
- **用输出识别更合理**：每个输出 tensor 都来自一次唯一的函数调用。换言之，一旦确定了输出 tensor，就能唯一确定它对应的 FunctionNode。所以上述 `torch.relu` 的例子会产生两个独立分支，符合直觉。

此外，当我们隐藏 inner tensors 时，用来识别的“输出”不一定是 TensorNode。上面的论证同样适用于输出为 ModuleNode 或 FunctionNode 的情况：输入识别不可靠，而输出识别能唯一确定对应的计算节点。

3. **ModuleNode**

ModuleNode 分为两种情况：**无状态模块（Stateless）** 与 **有状态模块（Stateful）**。

- **无状态模块（Stateless Module）**：指不包含 `torch.nn.parameter.Parameter` 的模块。它们通常只创建一次，但会在不同位置被重复调用多次，因此行为更像“函数”。所以它们的识别方式与 FunctionNode 相同（按“输出关联信息”来识别）。否则如果按 Python 对象 id 识别，所有 ReLU 之类的模块可能会被连到同一个节点上，图会非常难看。

- **有状态模块（Stateful Module）**：这类模块按 Python 对象（ModuleNode）的 id 来识别。这也很自然：当我们进行 rolling 折叠时，希望“拥有相同参数的同一个模块对象”在图中呈现为同一个节点。


