from typing import Callable, Any
from packaging import version

import pytest

import torchvision

from torchvision import __version__ as torchvision_version


from torchview import draw_graph
from tests.fixtures.u_net import UNet2  # type: ignore[attr-defined]
from tests.fixtures.dense_net import DenseNet  # type: ignore[attr-defined]
from tests.fixtures.ldc import LDC

# pass verify_result fixture to test functions
# so it can be used as a wrapper function


def test_simple_u_net_model(verify_result: Callable[..., Any]) -> None:
    model = UNet2(1, 1, use_bn=True, residual=True)
    model_graph = draw_graph(
        model, input_size=(1, 1, 28, 28),
        graph_name='UNet2',
    )

    verify_result([model_graph])


def test_expand_nested_u_net_model(verify_result: Callable[..., Any]) -> None:
    model = UNet2(1, 1, use_bn=True, residual=True)
    model_graph = draw_graph(
        model, input_size=(1, 1, 28, 28),
        graph_name='UNet2',
        expand_nested=True
    )

    verify_result([model_graph])


def test_simple_resnet_model(verify_result: Callable[..., Any]) -> None:

    model_graph = draw_graph(
        torchvision.models.resnet50(),
        input_size=(1, 3, 224, 224),
        graph_name='Resnet',
        depth=5
    )

    verify_result([model_graph])


def test_expand_nested_resnet_model(verify_result: Callable[..., Any]) -> None:

    model_graph = draw_graph(
        torchvision.models.resnet50(),
        input_size=(1, 3, 224, 224),
        graph_name='Resnet',
        expand_nested=True,
        depth=5
    )

    verify_result([model_graph])


def test_google(verify_result: Callable[..., Any]) -> None:
    model_graph1 = draw_graph(
        torchvision.models.GoogLeNet(init_weights=False),
        input_size=(1, 3, 224, 224),
        graph_name='GoogLeNet',
        depth=5
    )

    model_graph2 = draw_graph(
        torchvision.models.GoogLeNet(init_weights=False),
        input_size=(1, 3, 224, 224),
        graph_name='GoogLeNet',
        depth=5,
        mode='train'
    )

    verify_result([model_graph1, model_graph2])


def test_simple_densenet_model(verify_result: Callable[..., Any]) -> None:

    # needed depth=2, otherwise denseblock with 24
    # conv units will have 24*23/2 edges, too crowded image
    model_graph = draw_graph(
        torchvision.models.densenet121(),
        input_size=(1, 3, 224, 224),
        graph_name='DenseNet121',
        depth=2
    )

    verify_result([model_graph])


def test_custom_densenet_model(verify_result: Callable[..., Any]) -> None:

    config = DenseNet.get_default_config()
    model1 = DenseNet(config)
    config.efficient = False
    model2 = DenseNet(config)
    model_graph1 = draw_graph(
        model1, input_size=(1, 3, 224, 224),
        graph_name='CustomDenseNet',
        depth=3
    )

    model_graph2 = draw_graph(
        model2, input_size=(1, 3, 224, 224),
        graph_name='CustomDenseNet',
        depth=3
    )

    verify_result([model_graph1, model_graph2])

    assert (
        model_graph1.visual_graph.source == model_graph2.visual_graph.source
    ), 'Forward prop should not be affected by checkpoint propagation'


@pytest.mark.skipif(
    version.parse(torchvision_version) < version.parse('0.12.0'),
    reason=f"Torchvision version {torchvision_version} doesn't have this model."
)
def test_simple_ViT_model(verify_result: Callable[..., Any]) -> None:

    # needed depth=2, otherwise denseblock with 24
    # conv units will have 24*23/2 edges, too crowded image
    model_graph = draw_graph(
        torchvision.models.vit_b_16(),
        input_size=(1, 3, 224, 224),
        graph_name='ViT_b_16',
        depth=4
    )

    verify_result([model_graph])


def test_ldc_model(verify_result: Callable[..., Any]) -> None:

    # needed depth=2, otherwise denseblock with 24
    # conv units will have 24*23/2 edges, too crowded image

    model1 = LDC()
    model_graph1 = draw_graph(
        model1,
        input_size=(1, 3, 32, 32),
        graph_name='LDC',
        depth=3
    )

    model_graph2 = draw_graph(
        model1,
        input_size=(1, 3, 32, 32),
        graph_name='LDC_expanded',
        depth=3,
        expand_nested=True,
    )

    verify_result([model_graph1, model_graph2])
