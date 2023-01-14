from typing import Callable, Any

import torch
from torch.jit import trace
from tests.fixtures.models import (
    MLP,
)
from torchview import draw_graph


def test_traced_mlp(verify_result: Callable[..., Any]) -> None:

    input_data = torch.rand(1, 128)
    traced_mlp = trace(MLP().forward, input_data)
    model_graph_1 = draw_graph(
        traced_mlp, input_data=input_data,
        graph_name='MLP',
    )

    verify_result([model_graph_1])
