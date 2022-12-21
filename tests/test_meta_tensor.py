import sys

from typing import Callable, Any
from packaging import version

import pytest

import torch
import torchtext
import torchvision

from torch import nn
from torch import __version__ as torch_version
from torchtext import __version__ as torchtext_version
from torchvision import __version__ as torchvision_version

try:
    import transformers
except ImportError:
    pass

from tests.fixtures.models import (
    IdentityModel,
    MultipleInputNetDifferentDtypes,
    LSTMNet,
    RecursiveNet,
    SimpleRNN,
    SiameseNets,
    RecursiveRelu,
    OutputReused,
    InputNotUsed,
)

from tests.fixtures.custom_attention import (  # type: ignore[attr-defined]
    get_default_cfg, Block
)

from tests.fixtures.u_net import UNet2  # type: ignore[attr-defined]
from tests.fixtures.dense_net import DenseNet  # type: ignore[attr-defined]
from tests.fixtures.ldc import LDC

from torchview import draw_graph


if version.parse(torch_version) < version.parse('1.13.0'):
    pytest.skip("Meta tensor is not support for this version", allow_module_level=True)


DEVICE = 'meta'
TRANSFORMERS_MODULE = 'transformers'


def test_meta_identity_model(verify_result: Callable[..., Any]) -> None:
    model = IdentityModel()
    input_tensor = torch.rand(4, 4)
    model_graph = draw_graph(
        model, input_tensor,
        graph_name='IdentityModel',
        device=DEVICE,
    )

    verify_result([model_graph])


def test_meta_dict_input(verify_result: Callable[..., Any]) -> None:
    # TODO: expand this test to handle intermediate dict layers.
    model = MultipleInputNetDifferentDtypes()
    input_data = torch.randn(1, 300)
    other_input_data = torch.randn(1, 300).long()

    model_graph = draw_graph(
        model, input_data={"x1": input_data, "x2": other_input_data},
        device=DEVICE,
    )
    verify_result([model_graph])


def test_meta_simple_OutputReused(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        OutputReused(), input_size=[(1, 128), (1, 2), (1, 2), (1, 64)],
        graph_name='OutputReused',
        expand_nested=True,
        device=DEVICE,
    )

    verify_result([model_graph_1])


def test_meta_simple_InputNotUsed(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        InputNotUsed(), input_size=[(1, 128), (1, 2), (1, 2), (1, 64)],
        graph_name='OutputNotUsed',
        expand_nested=True,
        device=DEVICE,
    )

    verify_result([model_graph_1])


def test_meta_LSTM(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        LSTMNet(), input_size=(1, 100),
        graph_name='LSTM',
        dtypes=[torch.long],
        device=DEVICE,
    )

    verify_result([model_graph_1])


def test_meta_recursive_net(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        RecursiveNet(), input_size=(1, 64, 28, 28),
        graph_name='RecursiveNet',
        roll=True,
        device=DEVICE,
    )

    verify_result([model_graph_1])


def test_meta_roll_recursive_net(verify_result: Callable[..., Any]) -> None:

    model_graph_1 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        roll=True,
        device=DEVICE,
    )

    model_graph_2 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        roll=False,
        device=DEVICE,
    )

    verify_result([model_graph_1, model_graph_2])


def test_meta_expand_nested_recursive_net(verify_result: Callable[..., Any]) -> None:

    model_graph_1 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        expand_nested=True,
        roll=True,
        device=DEVICE,
    )

    model_graph_2 = draw_graph(
        SimpleRNN(), input_size=(2, 3),
        graph_name='RecursiveNet',
        expand_nested=True,
        roll=False,
        device=DEVICE,
    )

    verify_result([model_graph_1, model_graph_2])


def test_meta_explicit_siamese_nets(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        SiameseNets(), input_size=[(1, 1, 88, 88), (1, 1, 88, 88)],
        graph_name='SiameseNets',
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=True,
        depth=float('inf'),
        device=DEVICE,
    )

    verify_result([model_graph_1])


def test_meta_roll_recursive_relu(verify_result: Callable[..., Any]) -> None:
    model_graph_1 = draw_graph(
        RecursiveRelu(), input_size=(1, 2),
        graph_name='RecursiveRelu',
        device=DEVICE,
    )

    model_graph_2 = draw_graph(
        RecursiveRelu(), input_size=(1, 2),
        graph_name='RecursiveRelu',
        roll=True,
        device=DEVICE,
    )

    verify_result([model_graph_1, model_graph_2])


def test_meta_reusing_activation_layers(verify_result: Callable[..., Any]) -> None:
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
        device=DEVICE,
    )

    model_graph_2 = draw_graph(
        model2, input_size=(1, 2),
        graph_name='LeakySequential',
        device=DEVICE,
    )

    verify_result([model_graph_1, model_graph_2])


@pytest.mark.skipif(
    version.parse(torchtext_version) < version.parse('0.12.0'),
    reason=f"Torchtext version {torchtext_version} doesn't have this model."
)
def test_meta_bert_model(verify_result: Callable[..., Any]) -> None:
    xlmr_base = torchtext.models.ROBERTA_BASE_ENCODER
    model = xlmr_base.get_model(load_weights=False)
    transform = xlmr_base.transform()
    input_batch = ["Hello world", "How are you!"]
    model_input = torchtext.functional.to_tensor(
        transform(input_batch), padding_value=1)

    model_graph = draw_graph(
        model, model_input,
        graph_name='Roberta',
        depth=4,
        device=DEVICE,
    )

    verify_result([model_graph])


def test_meta_custom_attention(verify_result: Callable[..., Any]) -> None:

    config = get_default_cfg()
    model = Block(config=config)

    model_graph = draw_graph(
        model, input_size=(7, 2, 128),
        graph_name='custom-attention',
        hide_inner_tensors=False,
        hide_module_functions=False,
        depth=float('inf'),
        device=DEVICE,
    )

    verify_result([model_graph])


def test_meta_u_net_model(verify_result: Callable[..., Any]) -> None:
    model = UNet2(1, 1, use_bn=True, residual=True)
    model_graph = draw_graph(
        model, input_size=(1, 1, 28, 28),
        graph_name='UNet2',
        device=DEVICE,
    )

    verify_result([model_graph])


def test_meta_expand_nested_u_net_model(verify_result: Callable[..., Any]) -> None:
    model = UNet2(1, 1, use_bn=True, residual=True)
    model_graph = draw_graph(
        model, input_size=(1, 1, 28, 28),
        graph_name='UNet2',
        expand_nested=True,
        device=DEVICE,
    )

    verify_result([model_graph])


def test_meta_resnet_model(verify_result: Callable[..., Any]) -> None:

    model_graph = draw_graph(
        torchvision.models.resnet50(),
        input_size=(1, 3, 224, 224),
        graph_name='Resnet',
        depth=5,
        device=DEVICE,
    )

    verify_result([model_graph])


def test_meta_google(verify_result: Callable[..., Any]) -> None:
    model_graph1 = draw_graph(
        torchvision.models.GoogLeNet(init_weights=False),
        input_size=(1, 3, 224, 224),
        graph_name='GoogLeNet',
        depth=5,
        device=DEVICE,
    )

    model_graph2 = draw_graph(
        torchvision.models.GoogLeNet(init_weights=False),
        input_size=(1, 3, 224, 224),
        graph_name='GoogLeNet',
        depth=5,
        mode='train',
        device=DEVICE,
    )

    verify_result([model_graph1, model_graph2])


def test_meta_custom_densenet_model(verify_result: Callable[..., Any]) -> None:

    config = DenseNet.get_default_config()
    model1 = DenseNet(config)
    config.efficient = False
    model2 = DenseNet(config)
    model_graph1 = draw_graph(
        model1, input_size=(1, 3, 224, 224),
        graph_name='CustomDenseNet',
        depth=3,
        device=DEVICE,
    )

    model_graph2 = draw_graph(
        model2, input_size=(1, 3, 224, 224),
        graph_name='CustomDenseNet',
        depth=3,
        device=DEVICE,
    )

    verify_result([model_graph1, model_graph2])

    assert (
        model_graph1.visual_graph.source == model_graph2.visual_graph.source
    ), 'Forward prop should not be affected by checkpoint propagation'


@pytest.mark.skipif(
    version.parse(torchvision_version) < version.parse('0.12.0'),
    reason=f"Torchvision version {torchvision_version} doesn't have this model."
)
def test_meta_ViT_model(verify_result: Callable[..., Any]) -> None:

    # needed depth=2, otherwise denseblock with 24
    # conv units will have 24*23/2 edges, too crowded image
    model_graph = draw_graph(
        torchvision.models.vit_b_16(),
        input_size=(1, 3, 224, 224),
        graph_name='ViT_b_16',
        depth=4,
        device=DEVICE,
    )

    verify_result([model_graph])


def test_meta_ldc_model(verify_result: Callable[..., Any]) -> None:

    # needed depth=2, otherwise denseblock with 24
    # conv units will have 24*23/2 edges, too crowded image

    model1 = LDC()
    model_graph1 = draw_graph(
        model1,
        input_size=(1, 3, 32, 32),
        graph_name='LDC',
        depth=3,
        device=DEVICE,
    )

    model_graph2 = draw_graph(
        model1,
        input_size=(1, 3, 32, 32),
        graph_name='LDC_expanded',
        depth=3,
        expand_nested=True,
        device=DEVICE,
    )

    verify_result([model_graph1, model_graph2])


@pytest.mark.skipif(
    TRANSFORMERS_MODULE not in sys.modules,
    reason=f"{TRANSFORMERS_MODULE} module is not installed."
)
def test_meta_transformer_gpt2(verify_result: Callable[..., Any]) -> None:

    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to('cpu')
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to('cpu')

    model_graph1 = draw_graph(
        model, input_data=inputs, graph_name='gpt2', depth=3, device=DEVICE,
    )

    model_graph2 = draw_graph(
        model, input_data=inputs, graph_name='gpt2', depth=3, expand_nested=True,
        device=DEVICE,
    )
    verify_result([model_graph1, model_graph2])


@pytest.mark.skipif(
    TRANSFORMERS_MODULE not in sys.modules,
    reason=f"{TRANSFORMERS_MODULE} module is not installed."
)
def test_meta_transformer_automodel(verify_result: Callable[..., Any]) -> None:

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "bert-base-cased-finetuned-mrpc"
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased-finetuned-mrpc"
    )

    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    # The tokenizer will automatically add any model specific
    # separators (i.e. <CLS> and <SEP>) and tokens to
    # the sequence, as well as compute the attention masks.
    paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
    not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

    model_graph1 = draw_graph(
        model, input_data=paraphrase, graph_name='bert', depth=3, device=DEVICE,
    )

    model_graph2 = draw_graph(
        model, input_data=not_paraphrase, graph_name='bert',
        depth=3, expand_nested=True, device=DEVICE,
    )
    verify_result([model_graph1, model_graph2])


@pytest.mark.skipif(
    TRANSFORMERS_MODULE not in sys.modules,
    reason=f"{TRANSFORMERS_MODULE} module is not installed."
)
def test_transformer_t5(verify_result: Callable[..., Any]) -> None:

    tokenizer = transformers.T5Tokenizer.from_pretrained("t5-small")
    model = transformers.T5ForConditionalGeneration.from_pretrained("t5-small")

    input_ids = tokenizer(
        "translate English to German: The house is wonderful.", return_tensors="pt"
    ).input_ids
    labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids
    input_data = {
        'input_ids': input_ids.to('meta'),
        'labels': labels.to('meta'),
    }

    model_graph = draw_graph(model, input_data=input_data, device='meta')
    verify_result([model_graph])
