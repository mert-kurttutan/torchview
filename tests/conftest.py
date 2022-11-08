import sys
from pathlib import Path
from typing import Iterable, Callable, Any

import pytest
import graphviz

from torchview import ComputationGraph


def pytest_addoption(parser: pytest.Parser) -> None:
    """This allows us to check for these params in sys.argv."""
    parser.addoption("--overwrite", action="store_true", default=False)
    parser.addoption("--no-output", action="store_true", default=False)


@pytest.fixture()
def verify_result(
    request: pytest.FixtureRequest
) -> Callable[..., Any]:

    def _func(graphs: Iterable[ComputationGraph]) -> None:
        if "--no-output" in sys.argv:
            return

        test_name = request.node.name.replace("test_", "")

        if test_name == "input_size_half_precision":
            return

        verify_output(graphs, f"tests/test_output/{test_name}.gv")
    return _func


def verify_output(graphs: Iterable[ComputationGraph], filename: str) -> None:
    """
    Utility function to ensure output matches file.
    If you are writing new tests, set overwrite_file=True to generate the
    new test_output file.
    """
    filepath = Path(filename)
    test_output = ''
    for model_graph in graphs:
        test_output += model_graph.visual_graph.source
    if not graphs and not filepath.exists():
        return
    if "--overwrite" in sys.argv:
        filepath.parent.mkdir(exist_ok=True)
        filepath.touch(exist_ok=True)
        filepath.write_text(
            test_output, encoding="utf-8"
        )
        graphviz.render(engine='dot', filepath=filepath, format='png')

    verify_output_str(test_output, filename)


def verify_output_str(output: str, filename: str) -> None:
    expected = Path(filename).read_text(encoding="utf-8")
    assert output == expected
