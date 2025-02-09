# torchview

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPI version](https://badge.fury.io/py/torchview.svg)](https://badge.fury.io/py/torchview)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/torchview)](https://anaconda.org/conda-forge/torchview)
[![Build Status](https://github.com/mert-kurttutan/torchview/actions/workflows/test.yml/badge.svg)](https://github.com/mert-kurttutan/torchview/actions/workflows/test.yml)
[![GitHub license](https://img.shields.io/github/license/mert-kurttutan/torchview)](https://github.com/mert-kurttutan/torchview/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/mert-kurttutan/torchview/branch/main/graph/badge.svg)](https://codecov.io/gh/mert-kurttutan/torchview)
[![Downloads](https://pepy.tech/badge/torchview)](https://pepy.tech/project/torchview)

Torchview provides visualization of pytorch models in the form of visual graphs. Visualization includes tensors, modules, torch.functions and info such as input/output shapes.

Pytorch version of `plot_model of keras` (and more)

Supports PyTorch versions $\geq$ 1.7.

## Useful features

![Useful Features](https://user-images.githubusercontent.com/88637659/213171745-7acf07df-6578-4a50-a106-1a7b368f8d6c.svg#gh-dark-mode-only)![Useful Features](https://user-images.githubusercontent.com/88637659/213173736-6e91724c-8de1-4568-9d52-297b4b5ff0d2.svg#gh-light-mode-only)


## Installation

First, you need to install graphviz,

```Bash
pip install graphviz
```

For python interface of graphiz to work, you need to have dot layout command working in your system. If it isn't already installed, I suggest you run the following depeding on your OS,

Debian-based Linux distro (e.g. Ubuntu):

```Bash
apt-get install graphviz
```

Windows:

```Bash
choco install graphviz
```

macOS

```Bash
brew install graphviz
```

see more details [here](https://graphviz.readthedocs.io/en/stable/manual.html)

Then, continue with installing torchview using pip

```Bash
pip install torchview
```

or if you want via conda

```Bash
conda install -c conda-forge torchview
```

or if you want most up-to-date version, install directly from repo

```Bash
pip install git+https://github.com/mert-kurttutan/torchview.git
```

## How To Use

```python
from torchview import draw_graph

model = MLP()
batch_size = 2
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(model, input_size=(batch_size, 128), device='meta')
model_graph.visual_graph
```

![output](https://user-images.githubusercontent.com/88637659/206028431-b114f48e-6307-4ff3-b31a-a74185eb61b5.png)

## Notebook Examples

For more examples, see colab notebooks below,

**Introduction Notebook:** [![Introduction](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mert-kurttutan/torchview/blob/main/docs/example_introduction.ipynb)

**Computer Vision Models:** [![Vision](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mert-kurttutan/torchview/blob/main/docs/example_vision.ipynb)

**NLP Models:** [![NLP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mert-kurttutan/torchview/blob/main/docs/example_text.ipynb)

<!-- single_input_all_cols.out -->

**Note:** Output graphviz visuals return images with desired sizes. But sometimes, on VSCode, some shapes are being cropped due to large size and svg rendering on by VSCode. To solve this, I suggest you run the following

```python
import graphviz
graphviz.set_jupyter_format('png')
```

This problem does not occur on other platforms e.g. JupyterLab or Google Colab.

## Supported Features

* Almost all the models, RNN, Sequentials, Skip Connection, Hugging Face Models
* Support for meta tensors -> no memory consumption (for very Large models) (pytorch version $\geq$ 1.13) .
* Shows operations between tensors (in addition to module calls)
* Rolling/Unrolling feature. Recursively used modules can be rolled visually, see below.
* Diverse set of inputs/output types, e.g. nested data structure (dict, list, etc), huggingface tokenizer outputs

## Documentation

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
    '''Returns visual representation of the input Pytorch Module with
    ComputationGraph object. ComputationGraph object contains:

    1) Root nodes (usually tensor node for input tensors) which connect to all
    the other nodes of computation graph of pytorch module recorded during forward
    propagation.

    2) graphviz.Digraph object that contains visual representation of computation
    graph of pytorch module. This graph visual shows modules/ module hierarchy,
    torch_functions, shapes and tensors recorded during forward prop, for examples
    see documentation, and colab notebooks.


    Args:
        model (nn.Module):
            Pytorch model to represent visually.

        input_data (data structure containing torch.Tensor):
            input for forward method of model. Wrap it in a list for
            multiple args or in a dict or kwargs

        input_size (Sequence of Sizes):
            Shape of input data as a List/Tuple/torch.Size
            (dtypes must match model input, default is FloatTensors).
            Default: None

        graph_name (str):
            Name for graphviz.Digraph object. Also default name graphviz file
            of Graph Visualization
            Default: 'model'

        depth (int):
            Upper limit for depth of nodes to be shown in visualization.
            Depth is measured how far is module/tensor inside the module hierarchy.
            For instance, main module has depth=0, whereas submodule of main module
            has depth=1, and so on.
            Default: 3

        device (str or torch.device):
            Device to place and input tensors. Defaults to
            gpu if cuda is seen by pytorch, otherwise to cpu.
            Default: None

        dtypes (list of torch.dtype):
            Uses dtypes to set the types of input tensor if
            input size is given.

        mode (str):
            Mode of model to use for forward prop. Defaults
            to Eval mode if not given
            Default: None

        strict (bool):
            if true, graphviz visual does not allow multiple edges
            between nodes. Mutiple edge occurs e.g. when there are tensors
            from module node to module node and hiding those tensors
            Default: True
        
        expand_nested(bool):
            if true shows nested modules with dashed borders

        graph_dir (str):
            Sets the direction of visual graph
            'TB' -> Top to Bottom
            'LR' -> Left to Right
            'BT' -> Bottom to Top
            'RL' -> Right to Left
            Default: None -> TB

        hide_module_function (bool):
            Determines whether to hide module torch_functions. Some
            modules consist only of torch_functions (no submodule),
            e.g. nn.Conv2d.
            True => Dont include module functions in graphviz
            False => Include modules function in graphviz
            Default: True

        hide_inner_tensors (bool):
            Inner tensor is all the tensors of computation graph
            but input and output tensors
            True => Does not show inner tensors in graphviz
            False => Shows inner tensors in graphviz
            Default: True

        roll (bool):
            If true, rolls recursive modules.
            Default: False

        show_shapes (bool):
            True => Show shape of tensor, input, and output
            False => Dont show
            Default: True

        save_graph (bool):
            True => Saves output file of graphviz graph
            False => Does not save
            Default: False

        filename (str):
            name of the file to store dot syntax representation and
            image file of graphviz graph. Defaults to graph_name

        directory (str):
            directory in which to store graphviz output files.
            Default: .

    Returns:
        ComputationGraph object that contains visualization of the input
        pytorch model in the form of graphviz Digraph object
    '''
```

## Examples

### Rolled Version of Recursive Networks

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

### Show/Hide intermediate (hidden) tensors and Functionals

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


### ResNet / Skip Connection / Support for Torch operations / Nested Modules

```python
import torchvision

model_graph = draw_graph(resnet18(), input_size=(1,3,32,32), expand_nested=True)
model_graph.visual_graph
```

![expand_nested_resnet_model gv](https://user-images.githubusercontent.com/88637659/206036653-293f8ce7-04dd-4ac6-9de8-0061de505bba.png)

## TODO

* [ ] Display Module parameter info
* [ ] Support for Graph Neural Network (GNN)
* [ ] Support for Undirected edges for GNNs
* [ ] Support for torch-based functions[^1]

[^1]: Here, torch-based functions refers to any function that uses only torch functions and modules. This is more general than modules.

## Contributing

All issues and pull requests are much appreciated! If you are wondering how to build the project:

* torchview is actively developed using the latest version of Python.
* Changes should be backward compatible to Python 3.7, and will follow Python's End-of-Life guidance for old versions.
* Run `pip install -r requirements-dev.txt`. We use the latest versions of all dev packages.
* To run unit tests, run `pytest`.
* To update the expected output files, run `pytest --overwrite`.
* To skip output file tests, use `pytest --no-output`

## References

* Parts related to input processing and validation are taken/inspired from torchinfo repository!!.
* Many of the software related parts (e.g. testing) are also taken/inspired from torchinfo repository. So big thanks to @TylerYep!!!
* The algorithm for constructing visual graphs is thanks to `__torch_function__` and subclassing `torch.Tensor`. Big thanks to all those who developed this API!!.