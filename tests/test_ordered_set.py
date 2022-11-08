import pytest

from torchview.utils import OrderedSet


def test_discard_remove() -> None:

    oset_1: OrderedSet[int] = OrderedSet(range(10))

    oset_1.discard(7)
    assert 7 not in oset_1

    oset_1.discard(3)
    assert 3 not in oset_1


def test_equality() -> None:

    oset_1: OrderedSet[int] = OrderedSet(range(11))
    oset_1.remove(10)
    oset_2: OrderedSet[int] = OrderedSet(range(10))

    assert oset_1 == oset_2
    a_list = [1, 2, 3]

    with pytest.raises(NotImplementedError):
        _ = a_list == oset_1
