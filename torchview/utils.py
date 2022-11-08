from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar, MutableSet, Any, Generator

from torch.nn.parameter import Parameter

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrderedSet):
            raise NotImplementedError
        return len(self) == len(other) and list(self) == list(other)

    def issubset(self, other: OrderedSet[T]) -> bool:
        return self.map.keys() <= other.map.keys()


def is_generator_empty(parameters: Iterable[Parameter]) -> bool:
    try:
        _ = next(iter(parameters))
        return False
    except StopIteration:
        return True
