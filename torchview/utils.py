from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar, MutableSet, Any, Generator

from torch.nn.parameter import Parameter, UninitializedParameter
from torch import Tensor

T = TypeVar('T')


class OrderedSet(MutableSet[T]):
    '''Ordered set using mutableset. This is necessary for having reproducible
    iteration order. This property is useful for getting reproducible
    text representation of graphviz graphs, to be used in tests. This is
    because in algorith to produce graph many set objects are iterated.'''
    def __init__(self, iterable: Any | None = None) -> None:
        self.map: dict[T, None] = {}
        if iterable is not None:
            self |= iterable

    def __len__(self) -> int:
        return len(self.map)

    def __contains__(self, value: object) -> bool:
        return value in self.map

    def add(self, value: T) -> None:
        if value not in self.map:
            self.map[value] = None

    def remove(self, value: T) -> None:
        if value in self.map:
            _ = self.map.pop(value)

    def discard(self, value: T) -> None:
        if value in self.map:
            _ = self.map.pop(value)

    def __iter__(self) -> Generator[T, Any, Any]:
        for cur in self.map:
            yield cur

    def __repr__(self) -> str:
        if not self:
            return f'{self.__class__.__name__}'
        return f'{self.__class__.__name__}({list(self)})'


def is_generator_empty(parameters: Iterable[Parameter]) -> bool:
    try:
        _ = next(iter(parameters))
        return False
    except StopIteration:
        return True


def updated_dict(
    arg_dict: dict[str, Any], update_key: str, update_value: Any
) -> dict[str, Any]:
    return {
        keyword: value if keyword != update_key else update_value
        for keyword, value in arg_dict.items()
    }


def assert_input_type(
    func_name: str, valid_input_types: tuple[type, ...], in_var: Any
) -> None:

    assert isinstance(in_var, valid_input_types), (
        f'For an unknown reason, {func_name} function was '
        f'given input with wrong type. The input is of type: '
        f'{type(in_var)}. But, it should be {valid_input_types}'
    )


def stringify_attributes(
        obj, max_depth=3, current_depth=0, seen=None
) -> str:
    """Recursively create a one-line string representation of an object's attributes.

    - Handles dictionaries, class instances, lists, and tuples.
    - If `obj` is a `torch.Tensor`, only its shape and dtype are included.
    - Stops recursion at `max_depth`.
    """

    if current_depth > max_depth:
        return "..."

    if isinstance(obj, dict):
        return "{" + ", ".join(f"{k}: {stringify_attributes(v, max_depth, current_depth + 1, seen)}" for k, v in obj.items()) + "}"
    elif isinstance(obj, (list, tuple)):
        return "[" + ", ".join(stringify_attributes(v, max_depth, current_depth + 1, seen) for v in obj) + "]"
    elif isinstance(obj, Tensor):
        if isinstance(obj, UninitializedParameter):
            return "Tensor(<uninitialized>)"
        else:
            shape = Tensor.shape.__get__(obj)
            dtype = Tensor.dtype.__get__(obj)
            return f"Tensor(shape={tuple(shape)}, dtype={dtype})"
    elif hasattr(obj, "__dict__"):  # If it's a class instance
        attributes_limit = 20 if current_depth == 0 else 5 # Attributes are more interesting on the base level
        public_attributes = [(k, v) for k, v in vars(obj).items() if not k.startswith("_")]
        return f"{obj.__class__.__name__}(" + ", ".join(
            f"{k}={stringify_attributes(v, max_depth, current_depth + 1, seen)}"
            for k, v in public_attributes[:attributes_limit]
        ) + ("..." if len(public_attributes) > attributes_limit else "") + ")"
    else:
        return repr(obj)
