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


def test_transformer_automodel(verify_result: Callable[..., Any]) -> None:

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
        model, input_data=paraphrase, graph_name='bert', depth=3
    )

    model_graph2 = draw_graph(
        model, input_data=not_paraphrase, graph_name='bert', depth=3, expand_nested=True
    )
    verify_result([model_graph1, model_graph2])
