import pytest
import torch

from tests.fixtures.models import IdentityModel,  EdgeCaseModel
from torchview import draw_graph


def test_invalid_user_params() -> None:
    test = IdentityModel()
    with pytest.raises(RuntimeError):
        draw_graph(test, torch.randn(1, 28, 28), (1, 28, 28))

    with pytest.raises(ValueError):
        draw_graph(test, torch.randn(1, 28, 28), depth=-1)


def test_incorrect_model_forward() -> None:
    # Warning: these tests always raise RuntimeError.
    with pytest.raises(RuntimeError):
        draw_graph(EdgeCaseModel(throw_error=True), input_size=(5, 1, 28, 28))
    with pytest.raises(RuntimeError):
        draw_graph(
            EdgeCaseModel(throw_error=True), input_data=[[[torch.randn(1, 28, 28)]]]
        )


def test_input_size_possible_exceptions() -> None:
    test = IdentityModel()

    with pytest.raises(ValueError):
        draw_graph(test, input_size=[(3, 0)])
    with pytest.raises(TypeError):
        draw_graph(test, input_size={0: 1})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        draw_graph(test, input_size="hello")


def test_exception() -> None:
    input_size = (1, 1, 28, 28)
    draw_graph(EdgeCaseModel(throw_error=False), input_size=input_size)
    with pytest.raises(RuntimeError):
        draw_graph(EdgeCaseModel(throw_error=True), input_size=input_size)
