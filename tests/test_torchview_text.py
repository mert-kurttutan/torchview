from typing import Callable, Any
from packaging import version

import pytest
import torchtext

from torchtext import __version__ as torchtext_version

from tests.fixtures.custom_attention import (  # type: ignore[attr-defined]
    get_default_cfg, Block)

from torchview import draw_graph


@pytest.mark.skipif(
    version.parse(torchtext_version) < version.parse('0.12.0'),
    reason=f"Torchtext version {torchtext_version} doesn't have this model."
)
def test_simple_bert_model(verify_result: Callable[..., Any]) -> None:
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
    )

    verify_result([model_graph])


def test_simple_custom_attention(verify_result: Callable[..., Any]) -> None:

    config = get_default_cfg()
    model = Block(config=config)

    model_graph = draw_graph(
        model, input_size=(7, 2, 128),
        graph_name='custom-attention',
        hide_inner_tensors=False,
        hide_module_functions=False,
        depth=float('inf'),
    )

    verify_result([model_graph])
