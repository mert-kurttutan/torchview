from typing import Callable, Any

import pytest

from torchview import draw_graph

try:
    import transformers
except ImportError:
    pytest.skip("Transformers not Available", allow_module_level=True)


def test_transformer_gpt2(verify_result: Callable[..., Any]) -> None:

    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to('cpu')
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to('cpu')

    model_graph1 = draw_graph(
        model, input_data=inputs, graph_name='gpt2', depth=3
    )

    model_graph2 = draw_graph(
        model, input_data=inputs, graph_name='gpt2', depth=3, expand_nested=True
    )
    verify_result([model_graph1, model_graph2])
