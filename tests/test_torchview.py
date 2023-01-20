from os.path import exists
from typing import Callable, Any
from packaging import version

import pytest

import torch
from torch import nn
from torch import __version__ as torch_version
from tests.fixtures.models import (
    IdentityModel,
    MLP,
    SingleInputNet,
    MultipleInputNetDifferentDtypes,
    LSTMNet,
    RecursiveNet,
    SimpleRNN,
    SiameseNets,
    FunctionalNet,
    RecursiveRelu,
    ScalarNet,
    TowerBranches,
    OutputReused,
    InputNotUsed,
    CreateTensorsInside,
    EnsembleMLP,
    DetectionWrapper,
    IsolatedTensor,
)
from torchview import draw_graph

# pass verifty_result fixture to test functions
# so it can be used as a wrapper function


def test_simple_identity_model(verify_result: Callable[..., Any]) -> None:
    model = IdentityModel()
    input_tensor = torch.rand(4, 4)
    model_graph = draw_graph(
        model, input_tensor,
        graph_name='IdentityModel',
    )

    verify_result([model_graph])


def test_save_output_identity_model() -> None:
    model = IdentityModel()
    input_tensor = torch.rand(4, 4)
    _ = draw_graph(
        model, input_tensor,
        graph_name='save_output_identity_model',
        directory='tests/test_output',
        save_graph=True,
    )
    dot_file_path = 'tests/test_output/save_output_identity_model'
    assert exists(f'{dot_file_path}.gv') and exists(f'{dot_file_path}.gv.png')


def test_simple_single_input_net(verify_result: Callable[..., Any]) -> None:
    model = SingleInputNet()
    model_graph = draw_graph(
        model, input_size=(2, 1, 28, 28),
        graph_name='SingleInputNet',
    )

    verify_result([model_graph])


def test_fully_scalar_net(verify_result: Callable[..., Any]) -> None:
    model = ScalarNet()
    model_graph = draw_graph(
        model, input_data=[torch.randn(2, 64, 3, 3), 5],
        graph_name='SingleInputNet',
    )

    verify_result([model_graph])


def test_dict_input(verify_result: Callable[..., Any]) -> None:
    # TODO: expand this test to handle intermediate dict layers.
    model = MultipleInputNetDifferentDtypes()
    input_data = torch.randn(1, 300)
    other_input_data = torch.randn(1, 300).long()

    model_graph = draw_graph(
        model, input_data={"x1": input_data, "x2": other_input_data}
    )
    verify_result([model_graph])


def test_simple_MLP(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        MLP(), input_size=(1, 128),
        graph_name='MLP',
    )

    verify_result([model_graph_1])


def test_simple_OutputReused(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        OutputReused(), input_size=[(1, 128), (1, 2), (1, 2), (1, 64)],
        graph_name='OutputReused',
        expand_nested=True,
    )

    verify_result([model_graph_1])


def test_simple_InputNotUsed(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        InputNotUsed(), input_size=[(1, 128), (1, 2), (1, 2), (1, 64)],
        graph_name='OutputNotUsed',
        expand_nested=True,
    )

    verify_result([model_graph_1])


def test_inplace_MLP(verify_result: Callable[..., Any]) -> None:
    input_tensor = torch.rand(1, 128)
    model_graph_1 = draw_graph(
        MLP(inplace=True), input_tensor,
        graph_name='MLP',
    )

    model_graph_2 = draw_graph(
        MLP(inplace=False), input_tensor,
        graph_name='MLP',
    )

    verify_result([model_graph_1, model_graph_2])

    assert (
        model_graph_1.visual_graph.source == model_graph_2.visual_graph.source
    )


def test_simple_LSTM(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        LSTMNet(), input_size=(1, 100),
        graph_name='LSTM',
        dtypes=[torch.long],
    )

    verify_result([model_graph_1])


def test_simple_recursive_net(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        RecursiveNet(), input_size=(1, 64, 28, 28),
        graph_name='RecursiveNet',
        roll=True,
    )

    verify_result([model_graph_1])


def test_roll_recursive_net(verify_result: Callable[..., Any]) -> None:

    model_graph_1 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        roll=True,
    )

    model_graph_2 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        roll=False,
    )

    verify_result([model_graph_1, model_graph_2])


def test_expand_nested_recursive_net(verify_result: Callable[..., Any]) -> None:

    model_graph_1 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        expand_nested=True,
        roll=True,
    )

    model_graph_2 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        expand_nested=True,
        roll=False,
    )

    verify_result([model_graph_1, model_graph_2])


def test_expand_nested_explicit_recursive_net(
    verify_result: Callable[..., Any]
) -> None:

    model_graph_1 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        expand_nested=True,
        hide_module_functions=False,
        roll=True,
    )

    model_graph_2 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        expand_nested=True,
        hide_module_functions=False,
        roll=False,
    )

    verify_result([model_graph_1, model_graph_2])


def test_inplace_recursive_net(verify_result: Callable[..., Any]) -> None:

    model_graph_1 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
    )

    model_graph_2 = draw_graph(
        SimpleRNN(inplace=False), input_size=(2, 3),
        graph_name='RecursiveNet',
    )

    verify_result([model_graph_1, model_graph_2])

    assert (
        model_graph_1.visual_graph.source == model_graph_2.visual_graph.source
    )


def test_simple_siamese_nets(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        SiameseNets(), input_size=[(1, 1, 88, 88), (1, 1, 88, 88)],
        graph_name='SiameseNets',
    )

    verify_result([model_graph_1])


def test_explicit_siamese_nets(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        SiameseNets(), input_size=[(1, 1, 88, 88), (1, 1, 88, 88)],
        graph_name='SiameseNets',
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=True,
        depth=float('inf'),
    )

    verify_result([model_graph_1])


def test_no_shape_siamese_nets(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        SiameseNets(), input_size=[(1, 1, 88, 88), (1, 1, 88, 88)],
        graph_name='SiameseNets',
        hide_module_functions=False,
        hide_inner_tensors=False,
        show_shapes=False,
        depth=float('inf'),
    )

    verify_result([model_graph_1])


def test_simple_functional_net(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        FunctionalNet(), input_size=(1, 1, 32, 32),
        graph_name='FunctionalNet',
    )

    verify_result([model_graph_1])


def test_roll_recursive_relu(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        RecursiveRelu(), input_size=(1, 2),
        graph_name='RecursiveRelu',
    )

    model_graph_2 = draw_graph(
        RecursiveRelu(), input_size=(1, 2),
        graph_name='RecursiveRelu',
        roll=True,
    )

    verify_result([model_graph_1, model_graph_2])


def test_reusing_activation_layers(verify_result: Callable[..., Any]) -> None:
    act = nn.LeakyReLU(inplace=True)
    model1 = nn.Sequential(act, nn.Identity(), act, nn.Identity(), act)
    model2 = nn.Sequential(
        nn.LeakyReLU(inplace=True),
        nn.Identity(),
        nn.LeakyReLU(inplace=True),
        nn.Identity(),
        nn.LeakyReLU(inplace=True),
    )

    model_graph_1 = draw_graph(
        model1, input_size=(1, 2),
        graph_name='LeakySequential',
    )

    model_graph_2 = draw_graph(
        model2, input_size=(1, 2),
        graph_name='LeakySequential',
    )

    verify_result([model_graph_1, model_graph_2])


@pytest.mark.skipif(
    version.parse(torch_version) < version.parse('1.8'),
    reason=f"Torch version {torch_version} doesn't have this model."
)
def test_expand_nested_tower(verify_result: Callable[..., Any]) -> None:
    model = TowerBranches()
    model_graph = draw_graph(model, input_size=[(1, 10)], expand_nested=True)

    verify_result([model_graph])


def test_simple_create_tensor_inside(verify_result: Callable[..., Any]) -> None:
    model = CreateTensorsInside()
    model_graph = draw_graph(
        model,
        input_size=(1, 10),
        graph_name='CreateTensorInside',
        hide_inner_tensors=False,
        device='cpu',
        expand_nested=True,
    )

    verify_result([model_graph])


def test_simple_graph_dir(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        MLP(), input_size=(1, 128),
        graph_name='MLP',
        graph_dir='LR'
    )
    verify_result([model_graph_1])


def test_simple_ensemble(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        EnsembleMLP(), input_size=(1, 128), depth=1,
        graph_name='EnsembleMLP',
        graph_dir='LR',
        device='cpu'
    )
    verify_result([model_graph_1])


def test_input_inplace_changed(verify_result: Callable[..., Any]) -> None:
    input_data = [
        torch.rand(1, 192, 44, 44),
        torch.rand(1, 192, 44, 44),
    ]
    new_model = DetectionWrapper()
    model_graph = draw_graph(
        new_model,
        input_data=input_data,
        expand_nested=True,
        depth=3,
        graph_name='InplaceListInput',
        device='cpu'
    )
    verify_result([model_graph])


def test_isolated_tensor(verify_result: Callable[..., Any]) -> None:
    input_data = [
        torch.rand(1, 32),
    ]
    model = IsolatedTensor()
    model_graph = draw_graph(
        model,
        input_data=input_data,
        expand_nested=True,
        depth=3,
        graph_name='IsolatedTensor',
        device='cpu'
    )

    verify_result([model_graph])
