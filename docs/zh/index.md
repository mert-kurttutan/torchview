# torchview

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPI version](https://badge.fury.io/py/torchview.svg)](https://badge.fury.io/py/torchview)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/torchview)](https://anaconda.org/conda-forge/torchview)
[![Build Status](https://github.com/mert-kurttutan/torchview/actions/workflows/test.yml/badge.svg)](https://github.com/mert-kurttutan/torchview/actions/workflows/test.yml)
[![GitHub license](https://img.shields.io/github/license/mert-kurttutan/torchview)](https://github.com/mert-kurttutan/torchview/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/mert-kurttutan/torchview/branch/main/graph/badge.svg)](https://codecov.io/gh/mert-kurttutan/torchview/branch/main)
[![Downloads](https://pepy.tech/badge/torchview)](https://pepy.tech/project/torchview)

![banner](images/banner.png)

Torchview 用于将 PyTorch 模型可视化为**计算图**（visual graph）。可视化内容包括：tensor、module、`torch` 函数调用，以及输入/输出 shape 等信息。

它可以理解为 PyTorch 版本的 Keras `plot_model`（并且支持更多细节）。

支持 PyTorch 版本：\(\geq 1.7\)。

## 主要特性

![Useful Features](https://user-images.githubusercontent.com/88637659/213171745-7acf07df-6578-4a50-a106-1a7b368f8d6c.svg#gh-dark-mode-only)![Useful Features](https://user-images.githubusercontent.com/88637659/213173736-6e91724c-8de1-4568-9d52-297b4b5ff0d2.svg#gh-light-mode-only)

## 安装

首先需要安装 Graphviz：

```Bash
pip install graphviz
```

为了让 Graphviz 的 Python 接口正常工作，你的系统中需要能够调用 `dot` 布局命令。如果尚未安装 Graphviz，建议按操作系统安装：

Debian 系 Linux（如 Ubuntu）：

```Bash
apt-get install graphviz
```

Windows：

```Bash
choco install graphviz
```

macOS：

```Bash
brew install graphviz
```

更多细节可参考：[Graphviz 文档](https://graphviz.readthedocs.io/en/stable/manual.html)

然后用 pip 安装 torchview：

```Bash
pip install torchview
```

或者用 conda：

```Bash
conda install -c conda-forge torchview
```

如果你想安装最新版本，可以直接从仓库安装：

```Bash
pip install git+https://github.com/mert-kurttutan/torchview.git
```

## 快速使用

```python
from torchview import draw_graph

model = MLP()
batch_size = 2
# device='meta' -> 可视化时不会消耗实际显存/内存（只做结构推导）
model_graph = draw_graph(model, input_size=(batch_size, 128), device='meta')
model_graph.visual_graph
```

![output](https://user-images.githubusercontent.com/88637659/206028431-b114f48e-6307-4ff3-b31a-a74185eb61b5.png)

## Notebook 示例

更多示例可参考下面的 Colab：

**入门介绍：** [![Introduction](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mert-kurttutan/torchview/blob/main/docs/example_introduction.ipynb)

**计算机视觉模型：** [![Vision](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mert-kurttutan/torchview/blob/main/docs/example_vision.ipynb)

**NLP 模型：** [![NLP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mert-kurttutan/torchview/blob/main/docs/example_text.ipynb)

**注意：** torchview 的 Graphviz 可视化会返回“适配尺寸”的图像。但在 VSCode 中，由于 SVG 渲染与画布尺寸的限制，较大的图可能出现裁切。可通过以下方式改用 PNG 渲染：

```python
import graphviz
graphviz.set_jupyter_format('png')
```

该问题在 JupyterLab、Google Colab 等平台通常不会出现。

## 支持的能力

* 支持多数常见模型：RNN、`Sequential`、跳连（Skip Connection）、Hugging Face 模型等
* 支持 Meta Tensor，可在可视化超大模型时做到几乎不消耗内存（PyTorch \(\geq 1.13\)）
* 除了 module 调用，还能显示 tensor 之间的算子操作
* 支持 Rolling/Unrolling：可将递归调用的模块在图上“折叠/展开”（见下方示例）
* 支持多种输入/输出类型：如嵌套结构（dict/list 等）、Hugging Face tokenizer 输出等

## API 文档

```python
def draw_graph(
    model: nn.Module,
    input_data: INPUT_DATA_TYPE | None = None,
    input_size: INPUT_SIZE_TYPE | None = None,
    graph_name: str = 'model',
    depth: int | float = 3,
    device: torch.device | str | None = None,
    dtypes: list[torch.dtype] | None = None,
    mode: str | None = None,
    strict: bool = True,
    expand_nested: bool = False,
    graph_dir: str | None = None,
    hide_module_functions: bool = True,
    hide_inner_tensors: bool = True,
    roll: bool = False,
    show_shapes: bool = True,
    save_graph: bool = False,
    filename: str | None = None,
    directory: str = '.',
    **kwargs: Any,
) -> ComputationGraph:
    '''返回输入 PyTorch Module 的可视化表示（ComputationGraph）。
    ComputationGraph 包含：

    1) 根节点（通常是输入 tensor 节点），它连接到 forward 过程中记录的所有其它节点

    2) `graphviz.Digraph` 对象，用于承载计算图的可视化表示。图中会展示 module/module 层级、
    torch 函数、shape，以及 forward 中记录到的 tensor。相关示例可见文档与 Colab notebook。

    Args:
        model (nn.Module):
            需要可视化的 PyTorch 模型。

        input_data (包含 torch.Tensor 的数据结构):
            作为模型 forward 的输入。多个位置参数可放入 list；
            或用 dict / kwargs 形式传入。

        input_size (shape 序列):
            输入数据的 shape（list/tuple/torch.Size）。如果给了 `input_size`，
            那么 `dtypes` 需要与模型输入一致（默认使用 FloatTensor）。
            Default: None

        graph_name (str):
            Graphviz `Digraph` 的名称，也会作为默认输出文件名。
            Default: 'model'

        depth (int):
            可视化中节点展示的最大深度。深度定义为：节点在模块层级中的“嵌套层数”。
            例如主模块 depth=0，主模块的子模块 depth=1，以此类推。
            Default: 3

        device (str or torch.device):
            放置输入 tensor 的 device。若未指定：
            - PyTorch 检测到 CUDA 则使用 GPU
            - 否则使用 CPU
            Default: None

        dtypes (list[torch.dtype]):
            当提供 `input_size` 时，用 `dtypes` 设置输入 tensor 的 dtype。

        mode (str):
            forward 传播时使用的模型模式；未指定则默认用 eval。
            Default: None

        strict (bool):
            如果为 true，则 Graphviz 可视化不允许同一对节点之间出现多条边。
            多条边可能发生在：module 节点之间同时存在 tensor 边，但你又选择隐藏这些 tensor 时。
            Default: True

        expand_nested(bool):
            如果为 true，则用虚线边框展示嵌套模块。

        graph_dir (str):
            设置图的方向：
            'TB' -> 从上到下
            'LR' -> 从左到右
            'BT' -> 从下到上
            'RL' -> 从右到左
            Default: None -> TB

        hide_module_function (bool):
            是否隐藏 module 内部的 torch function。部分模块只由 torch function 构成（无子模块），
            例如 `nn.Conv2d`。
            True => 不在图中展示 module functions
            False => 在图中展示 module functions
            Default: True

        hide_inner_tensors (bool):
            inner tensor 指除输入/输出外，在计算图内部流转的 tensor。
            True => 不展示 inner tensors
            False => 展示 inner tensors
            Default: True

        roll (bool):
            若为 true，则折叠递归模块（Rolling）。
            Default: False

        show_shapes (bool):
            True => 展示 tensor 的 shape（含输入/输出）
            False => 不展示 shape
            Default: True

        save_graph (bool):
            True => 保存 Graphviz 输出文件
            False => 不保存
            Default: False

        filename (str):
            保存 dot 语法与图像文件时使用的文件名；默认等于 graph_name。

        directory (str):
            保存 Graphviz 输出文件的目录。
            Default: .

    Returns:
        ComputationGraph：包含 Graphviz `Digraph` 的计算图对象。
    '''
```

## 示例

### 递归网络的折叠（Rolled Version）

```python
from torchview import draw_graph

model_graph = draw_graph(
    SimpleRNN(), input_size=(2, 3),
    graph_name='RecursiveNet',
    roll=True
)
model_graph.visual_graph
```

![rnns](https://user-images.githubusercontent.com/88637659/206644016-23a89c81-1d6a-4558-82f4-33f179b345f3.png)

### 显示/隐藏中间（hidden）tensor 与 functionals

```python
# Show inner tensors and Functionals
model_graph = draw_graph(
    MLP(), input_size=(2, 128),
    graph_name='MLP',
    hide_inner_tensors=False,
    hide_module_functions=False,
)

model_graph.visual_graph
```

![download](https://user-images.githubusercontent.com/88637659/206188796-4b9e57ef-8d33-469b-b8e0-2c47b06fe70b.png)

### ResNet / 跳连 / 支持 torch 运算 / 嵌套模块展示

```python
import torchvision

model_graph = draw_graph(resnet18(), input_size=(1,3,32,32), expand_nested=True)
model_graph.visual_graph
```

![expand_nested_resnet_model gv](https://user-images.githubusercontent.com/88637659/206036653-293f8ce7-04dd-4ac6-9de8-0061de505bba.png)

## TODO

* [ ] 展示 Module 的参数信息（parameter info）
* [ ] 支持图神经网络（GNN）
* [ ] 为 GNN 支持无向边
* [ ] 支持 torch-based functions[^1]

[^1]: 这里的 torch-based functions 指“只使用 torch 函数和模块实现的函数”。该概念比 module 更泛化。

## 贡献指南

中文版本由 @1985312383（GitHub）友情提供。

我们非常欢迎 issue 与 PR！如果你想了解如何构建本项目：

* torchview 使用最新 Python 版本进行活跃开发。
* 改动需要保持对 Python 3.7 的向后兼容，并遵循 Python 的旧版本生命周期策略。
* 运行 `pip install -r requirements-dev.txt` 安装开发依赖（我们使用最新的 dev 包版本）。
* 单测：运行 `pytest`
* 更新期望输出：运行 `pytest --overwrite`
* 跳过输出文件测试：运行 `pytest --no-output`

## 参考

* 输入处理与校验相关部分借鉴/参考了 torchinfo 仓库
* 软件工程相关部分（如测试）也借鉴了 torchinfo（感谢 @TylerYep）
* 计算图构建算法得益于 `__torch_function__` 与 `torch.Tensor` 的 subclass 机制（感谢相关贡献者）


