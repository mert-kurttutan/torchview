# 示例：入门介绍

本教程用于介绍 `torchview` 的 API、图中节点的含义，以及几个最常用的参数组合。

## 0. 安装

你需要 Graphviz（用于渲染最终图），以及 torchview：

```bash
pip install graphviz
pip install torchview
```

如果系统里还没有 `dot` 命令，请按你的操作系统安装 Graphviz（例如 Windows 可用 `choco install graphviz`）。

## 1. 第一个例子：MLP

下面是一个简单的多层感知机（MLP）：

```python
import torch
from torch import nn
from torchview import draw_graph


class MLP(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace),
            nn.Linear(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
```

用 `draw_graph` 生成可视化图：

```python
model_graph = draw_graph(
    MLP(),
    input_size=(2, 128),
    graph_name="MLP",
    hide_inner_tensors=False,
    hide_module_functions=False,
)
model_graph.visual_graph
```

这里我们把 `hide_inner_tensors=False`、`hide_module_functions=False` 打开，是为了让图展示更多细节（更适合学习）。

## 2. 图里有哪些节点？

torchview 的图里主要有三类节点：

- **Tensor Node**：表示张量（输入/中间/输出）
- **Function Node**：表示算子/函数调用（例如 `torch.relu`、`torch.add`）
- **Module Node**：表示模块调用（例如 `nn.Linear`、`nn.Sequential`）

节点标签通常包含：节点名、层级深度、输入/输出 shape 等信息。

## 3. Rolling：折叠递归模块

当模型中存在“重复调用同一个模块对象”（例如 RNN 的 cell 反复复用）时，图可能会非常长。

这时可以使用 `roll=True` 将递归结构**折叠**：

```python
import torch
from torch import nn


class SimpleRNN(nn.Module):
    """一个用于演示 rolling 的简化 RNN（使用 LSTMCell 反复迭代）"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.hid_dim = 2
        self.input_dim = 3
        self.max_length = 4
        self.lstm = nn.LSTMCell(self.input_dim, self.hid_dim)
        self.activation = nn.LeakyReLU(inplace=inplace)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        b_size = token_embedding.size()[0]
        hx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)
        cx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)

        for _ in range(self.max_length):
            hx, cx = self.lstm(token_embedding, (hx, cx))
            hx = self.activation(hx)

        return hx

model_graph = draw_graph(
    SimpleRNN(),
    input_size=(2, 3),
    graph_name="RecursiveNet",
    roll=True,
)
model_graph.visual_graph
```

折叠后，图中边旁边的数字表示该边在 forward 中被“重复使用”的次数；若次数为 1 则通常不会显示。

## 4. Resize：缩放输出图

如果渲染出来的图过大，可以缩放：

```python
model_graph.resize_graph(scale=0.5)
model_graph.visual_graph
```

## 5. 一个常见的小坑（VSCode 渲染裁切）

在 VSCode 的 Notebook 渲染里，较大的 SVG 图可能被裁切。可以改用 PNG：

```python
import graphviz
graphviz.set_jupyter_format("png")
```


